import argparse
import json
import os
from pathlib import Path

import torch
import wandb
from monai.utils import set_determinism

from VAE_finetune_utils.data import (
    build_dataset_records,
    create_dataloader,
    ensure_paths_exist,
    load_dataset_specs,
    sample_train_records_for_epoch,
)
from VAE_finetune_utils.trainer import VAEFinetuneTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune MAISI VAE on multi-dataset npy data.")
    parser.add_argument("--dataset_spec_json", type=str, required=True, help="List of dataset specs, each has name/train_json/val_json/samples_per_epoch.")
    parser.add_argument("--environment_file", type=str, default="./configs/environment_maisi_vae_train.json")
    parser.add_argument("--network_config", type=str, default="./configs/config_network_rflow.json")
    parser.add_argument("--train_config", type=str, default="./configs/config_maisi_vae_train.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/vae_finetune")
    parser.add_argument("--project", type=str, default="NV-Generate-CTMR-vae-finetune")
    parser.add_argument("--run_name", type=str, default="vae_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every_n_steps", type=int, default=200)
    parser.add_argument("--val_vis_num_images", type=int, default=4)
    parser.add_argument("--lr_warmup_epoch1", type=int, default=10)
    parser.add_argument("--lr_warmup_epoch2", type=int, default=20)
    parser.add_argument("--lr_warmup_scale1", type=float, default=0.01)
    parser.add_argument("--lr_warmup_scale2", type=float, default=0.1)
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def merge_json_into_args(args):
    for cfg_path in [args.environment_file, args.network_config, args.train_config]:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "data_option" in payload:
            payload = {**payload["data_option"], **payload["autoencoder_train"]}
        for k, v in payload.items():
            setattr(args, k, v)


def main():
    args = parse_args()
    merge_json_into_args(args)
    if args.disable_amp:
        args.amp = False

    set_determinism(seed=args.seed)

    specs = load_dataset_specs(args.dataset_spec_json)
    ensure_paths_exist(specs)
    train_by_dataset, val_records = build_dataset_records(specs)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.best_autoencoder_path = os.path.join(args.output_dir, "autoencoder_best.pt")
    args.latest_checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pt")

    if not hasattr(args, "trained_autoencoder_path"):
        args.trained_autoencoder_path = "models/autoencoder_v1.pt"
    if not hasattr(args, "finetune"):
        args.finetune = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    trainer = VAEFinetuneTrainer(args=args, device=device)
    trainer.load_pretrained_if_needed()

    dataloader_val = create_dataloader(
        records=val_records,
        patch_size=args.patch_size,
        batch_size=args.val_batch_size,
        is_train=False,
        num_workers=args.num_workers,
    )

    for epoch in range(args.n_epochs):
        epoch_records = sample_train_records_for_epoch(train_by_dataset, specs=specs, seed=args.seed + epoch)
        if len(epoch_records) == 0:
            raise RuntimeError("No training samples selected for this epoch. Please check dataset_spec_json.")

        dataloader_train = create_dataloader(
            records=epoch_records,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            is_train=True,
            num_workers=args.num_workers,
        )

        epoch_losses = trainer.train_one_epoch(dataloader_train=dataloader_train, dataloader_val=dataloader_val, epoch=epoch)
        wandb.log({"epoch": epoch, "train/selected_samples": len(epoch_records), **{f"train/{k}": v for k, v in epoch_losses.items()}}, step=trainer.state.global_step)


    final_val_loss = trainer.validate(dataloader_val, epoch=args.n_epochs)
    wandb.log({"final/val_loss": final_val_loss}, step=trainer.state.global_step)
    torch.save(trainer.autoencoder.state_dict(), os.path.join(args.output_dir, "autoencoder_final.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()
