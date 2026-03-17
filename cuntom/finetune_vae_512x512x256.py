#!/usr/bin/env python3
"""Finetune MAISI VAE on 512x512x256 data with DDP + Weights & Biases.

Recommended launch (2x H20):
torchrun --nproc_per_node=2 cuntom/finetune_vae_512x512x256.py --config /path/to/finetune_vae_512x512x256.json

Config schema (JSON):
{
  "seed": 42,
  "network_config": "configs/config_network_rflow.json",
  "train_config": "configs/config_maisi_vae_train.json",
  "pretrained_vae": "models/autoencoder_epoch273.pt",
  "resume_discriminator": null,
  "output_dir": "./runs/vae_finetune_512x512x256",
  "amp": true,
  "num_workers": 6,
  "save_every": 5,
  "wandb": {
    "enabled": true,
    "project": "maisi-vae-finetune",
    "name": "h20-512x512x256",
    "entity": null,
    "tags": ["vae", "finetune", "512x512x256"]
  },
  "data": {
    "train": [
      {
        "name": "ct_train_a",
        "modality": "ct",
        "data_root": "/path/ct_train_a",
        "datalist_json": "/path/ct_train_a.json",
        "datalist_key": "training"
      }
    ],
    "val": [
      {
        "name": "ct_val_a",
        "modality": "ct",
        "data_root": "/path/ct_val_a",
        "datalist_json": "/path/ct_val_a.json",
        "datalist_key": "validation"
      }
    ]
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DistributedSampler

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune MAISI VAE with DDP and wandb logging.")
    parser.add_argument("--config", required=True, help="Path to finetune config JSON.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_samples(groups: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for group in groups:
        modality = group["modality"].lower()
        if "datalist_json" in group:
            key = group.get("datalist_key", "training")
            group_samples = load_decathlon_datalist(
                group["datalist_json"], is_segmentation=False, data_list_key=key, base_dir=group.get("data_root")
            )
        else:
            group_samples = group["samples"]

        data_root = group.get("data_root")
        for item in group_samples:
            image_path = Path(item["image"])
            if (not image_path.is_absolute()) and data_root:
                image_path = Path(data_root) / image_path
            samples.append({"image": str(image_path), "class": modality})
    return samples


def _loss_weighted_sum(losses: dict[str, torch.Tensor], kl_weight: float, perceptual_weight: float) -> torch.Tensor:
    return losses["recons_loss"] + kl_weight * losses["kl_loss"] + perceptual_weight * losses["p_loss"]


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

    network_cfg = _load_json(cfg["network_config"])
    for k, v in network_cfg.items():
        setattr(args, k, v)
    train_cfg = _load_json(cfg["train_config"])
    for k, v in train_cfg["data_option"].items():
        setattr(args, k, v)
    for k, v in train_cfg["autoencoder_train"].items():
        setattr(args, k, v)

    train_files = _parse_samples(cfg["data"]["train"])
    val_files = _parse_samples(cfg["data"]["val"])

    train_transform = VAE_Transform(
        is_train=True,
        random_aug=bool(args.random_aug),
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        select_channel=int(args.select_channel),
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        select_channel=int(args.select_channel),
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transform,
        cache_rate=float(args.cache),
        num_workers=int(cfg.get("num_workers", 6)),
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transform,
        cache_rate=float(args.cache),
        num_workers=int(cfg.get("num_workers", 6)),
    )

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
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    autoencoder.load_state_dict(torch.load(cfg["pretrained_vae"], map_location="cpu"), strict=True)
    if cfg.get("resume_discriminator"):
        discriminator.load_state_dict(torch.load(cfg["resume_discriminator"], map_location="cpu"), strict=True)

    if is_ddp:
        autoencoder = DistributedDataParallel(autoencoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    amp_enabled = bool(cfg.get("amp", True))
    intensity_loss = MSELoss() if args.recon_loss == "l2" else L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if amp_enabled else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if amp_enabled else 1e-08)

    def warmup_rule(epoch: int) -> float:
        if epoch < 10:
            return 0.01
        if epoch < 20:
            return 0.1
        return 1.0

    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5, enabled=amp_enabled)
    scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5, enabled=amp_enabled)

    output_dir = Path(cfg.get("output_dir", "./runs/vae_finetune_512x512x256"))
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(cfg.get("wandb", {}).get("enabled", False)) and rank == 0
    if use_wandb:
        wandb.init(
            project=cfg["wandb"].get("project", "maisi-vae-finetune"),
            name=cfg["wandb"].get("name"),
            entity=cfg["wandb"].get("entity"),
            tags=cfg["wandb"].get("tags"),
            config=cfg,
        )

    val_inferer = (
        SlidingWindowInferer(
            roi_size=args.val_sliding_window_patch_size,
            sw_batch_size=1,
            progress=False,
            overlap=0.0,
            device=torch.device("cpu"),
            sw_device=device,
        )
        if args.val_sliding_window_patch_size
        else SimpleInferer()
    )

    best_val = float("inf")
    max_epochs = int(args.n_epochs)
    save_every = int(cfg.get("save_every", 5))

    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        autoencoder.train()
        discriminator.train()
        epoch_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0, "g_loss": 0.0, "d_loss": 0.0}

        for batch in train_loader:
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                loss_recons = intensity_loss(reconstruction.float(), images.float())
                loss_kl = KL_loss(z_mu, z_sigma)
                loss_p = perceptual_loss(reconstruction.float(), images.float())
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                loss_g_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                losses = {"recons_loss": loss_recons, "kl_loss": loss_kl, "p_loss": loss_p}
                loss_g = _loss_weighted_sum(losses, args.kl_weight, args.perceptual_weight) + args.adv_weight * loss_g_adv

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            with autocast("cuda", enabled=amp_enabled):
                logits_fake = discriminator(reconstruction.detach().contiguous().float())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.detach().contiguous().float())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = 0.5 * (loss_d_fake + loss_d_real)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            epoch_losses["recons_loss"] += loss_recons.item()
            epoch_losses["kl_loss"] += loss_kl.item()
            epoch_losses["p_loss"] += loss_p.item()
            epoch_losses["g_loss"] += loss_g.item()
            epoch_losses["d_loss"] += loss_d.item()

        n_train = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= max(n_train, 1)

        scheduler_g.step()
        scheduler_d.step()

        autoencoder.eval()
        val_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0}
        with torch.no_grad():
            for val_batch in val_loader:
                images = val_batch["image"].to(device).contiguous()
                with autocast("cuda", enabled=amp_enabled):
                    if isinstance(val_inferer, SlidingWindowInferer):
                        reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, autoencoder, images)
                    else:
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                    val_losses["recons_loss"] += intensity_loss(reconstruction.float(), images.float()).item()
                    val_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_losses["p_loss"] += perceptual_loss(reconstruction.float(), images.float()).item()

        n_val = len(val_loader)
        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        val_total = val_losses["recons_loss"] + args.kl_weight * val_losses["kl_loss"] + args.perceptual_weight * val_losses["p_loss"]

        if is_ddp:
            for tensor_dict in (epoch_losses, val_losses):
                for key, value in tensor_dict.items():
                    t = torch.tensor(value, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    tensor_dict[key] = (t / world_size).item()
            t_val_total = torch.tensor(val_total, device=device)
            dist.all_reduce(t_val_total, op=dist.ReduceOp.SUM)
            val_total = (t_val_total / world_size).item()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{max_epochs} train={epoch_losses} val={val_losses} val_total={val_total:.6f}")
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "lr": scheduler_g.get_last_lr()[0],
                        "train/recons_loss": epoch_losses["recons_loss"],
                        "train/kl_loss": epoch_losses["kl_loss"],
                        "train/perceptual_loss": epoch_losses["p_loss"],
                        "train/g_loss": epoch_losses["g_loss"],
                        "train/d_loss": epoch_losses["d_loss"],
                        "val/recons_loss": val_losses["recons_loss"],
                        "val/kl_loss": val_losses["kl_loss"],
                        "val/perceptual_loss": val_losses["p_loss"],
                        "val/total": val_total,
                    }
                )

            autoencoder_to_save = autoencoder.module if isinstance(autoencoder, DistributedDataParallel) else autoencoder
            discriminator_to_save = discriminator.module if isinstance(discriminator, DistributedDataParallel) else discriminator

            if (epoch + 1) % save_every == 0:
                torch.save(autoencoder_to_save.state_dict(), output_dir / f"autoencoder_epoch{epoch + 1}.pt")
                torch.save(discriminator_to_save.state_dict(), output_dir / f"discriminator_epoch{epoch + 1}.pt")

            if val_total < best_val:
                best_val = val_total
                torch.save(autoencoder_to_save.state_dict(), output_dir / "autoencoder_best.pt")
                torch.save(discriminator_to_save.state_dict(), output_dir / "discriminator_best.pt")

    if use_wandb:
        wandb.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
