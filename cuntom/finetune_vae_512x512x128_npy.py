#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler

from cuntom.npy_vae_data import NpyVaeDataset, load_paths
from scripts.utils import KL_loss, define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune 512x512x128 VAE on npy data.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.config)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed = int(cfg.get("seed", 42)) + rank
    set_determinism(seed=seed)
    torch.manual_seed(seed)

    net_cfg = _load_json(cfg["network_config"])
    train_cfg = _load_json(cfg["train_config"])
    for k, v in net_cfg.items():
        setattr(args, k, v)
    for k, v in train_cfg["autoencoder_train"].items():
        setattr(args, k, v)

    train_paths = load_paths(cfg["data"]["train_paths_json"], key=cfg["data"].get("train_key"), data_root=cfg["data"].get("data_root"))
    val_paths = load_paths(cfg["data"]["val_paths_json"], key=cfg["data"].get("val_key"), data_root=cfg["data"].get("data_root"))

    target_hw = int(cfg.get("target_hw", 512))
    target_depth = int(cfg.get("target_depth", 128))

    train_ds = NpyVaeDataset(train_paths, target_hw=target_hw, target_depth=target_depth, resize_if_smaller=True)
    val_ds = NpyVaeDataset(val_paths, target_hw=target_hw, target_depth=target_depth, resize_if_smaller=True)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 6)),
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.val_batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, int(cfg.get("num_workers", 6)) // 2),
        pin_memory=True,
    )

    args.autoencoder_def["num_splits"] = 1
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(cfg["pretrained_vae"], map_location="cpu"), strict=True)

    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    if is_ddp:
        autoencoder = DistributedDataParallel(autoencoder, device_ids=[local_rank], output_device=local_rank)
        discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank)

    amp_enabled = bool(cfg.get("amp", True))
    intensity_loss = MSELoss() if args.recon_loss == "l2" else L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr, eps=1e-06 if amp_enabled else 1e-08)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, eps=1e-06 if amp_enabled else 1e-08)
    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda epoch: 0.1 if epoch < 20 else 1.0)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambda epoch: 0.1 if epoch < 20 else 1.0)

    scaler_g = GradScaler("cuda", enabled=amp_enabled)
    scaler_d = GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(cfg.get("output_dir", "./runs/vae_finetune_512x512x128_npy"))
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(cfg.get("wandb", {}).get("enabled", False)) and rank == 0
    if use_wandb:
        wandb.init(project=cfg["wandb"].get("project", "maisi-vae-finetune"), name=cfg["wandb"].get("name"), config=cfg)

    best_val = float("inf")
    for epoch in range(int(args.n_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        autoencoder.train()
        discriminator.train()
        train_log = {"recons": 0.0, "kl": 0.0, "perceptual": 0.0, "g": 0.0, "d": 0.0}

        for batch in train_loader:
            x = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled):
                xrec, z_mu, z_sigma = autoencoder(x)
                l_rec = intensity_loss(xrec.float(), x.float())
                l_kl = KL_loss(z_mu, z_sigma)
                l_per = perceptual_loss(xrec.float(), x.float())
                logits_fake = discriminator(xrec.float())[-1]
                l_g_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                l_g = l_rec + args.kl_weight * l_kl + args.perceptual_weight * l_per + args.adv_weight * l_g_adv

            scaler_g.scale(l_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            with autocast("cuda", enabled=amp_enabled):
                logits_fake = discriminator(xrec.detach().float())[-1]
                logits_real = discriminator(x.detach().float())[-1]
                l_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                l_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                l_d = 0.5 * (l_d_fake + l_d_real)

            scaler_d.scale(l_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            train_log["recons"] += l_rec.item()
            train_log["kl"] += l_kl.item()
            train_log["perceptual"] += l_per.item()
            train_log["g"] += l_g.item()
            train_log["d"] += l_d.item()

        for k in train_log:
            train_log[k] /= max(1, len(train_loader))

        autoencoder.eval()
        val_log = {"recons": 0.0, "kl": 0.0, "perceptual": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                with autocast("cuda", enabled=amp_enabled):
                    xrec, z_mu, z_sigma = autoencoder(x)
                    val_log["recons"] += intensity_loss(xrec.float(), x.float()).item()
                    val_log["kl"] += KL_loss(z_mu, z_sigma).item()
                    val_log["perceptual"] += perceptual_loss(xrec.float(), x.float()).item()

        for k in val_log:
            val_log[k] /= max(1, len(val_loader))
        val_total = val_log["recons"] + args.kl_weight * val_log["kl"] + args.perceptual_weight * val_log["perceptual"]

        if is_ddp:
            t = torch.tensor(val_total, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_total = (t / world_size).item()

        if rank == 0:
            print(f"Epoch {epoch + 1}: train={train_log}, val={val_log}, val_total={val_total:.6f}")
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "lr": scheduler_g.get_last_lr()[0],
                        "train/recons": train_log["recons"],
                        "train/kl": train_log["kl"],
                        "train/perceptual": train_log["perceptual"],
                        "train/g": train_log["g"],
                        "train/d": train_log["d"],
                        "val/recons": val_log["recons"],
                        "val/kl": val_log["kl"],
                        "val/perceptual": val_log["perceptual"],
                        "val/total": val_total,
                    }
                )

            ae_to_save = autoencoder.module if isinstance(autoencoder, DistributedDataParallel) else autoencoder
            d_to_save = discriminator.module if isinstance(discriminator, DistributedDataParallel) else discriminator
            torch.save(ae_to_save.state_dict(), output_dir / "autoencoder_last.pt")
            torch.save(d_to_save.state_dict(), output_dir / "discriminator_last.pt")
            if val_total < best_val:
                best_val = val_total
                torch.save(ae_to_save.state_dict(), output_dir / "autoencoder_best.pt")
                torch.save(d_to_save.state_dict(), output_dir / "discriminator_best.pt")

        scheduler_g.step()
        scheduler_d.step()

    if use_wandb:
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
