#!/usr/bin/env python3
"""Evaluate MAISI VAE encode-decode performance on 512x512x256 volumes.

Config schema (JSON):
{
  "seed": 42,
  "network_config": "configs/config_network_rflow.json",
  "vae_checkpoint": "models/autoencoder_epoch273.pt",
  "output_dir": "./runs/vae_eval_512x512x256",
  "device": "cuda",
  "amp": true,
  "num_workers": 4,
  "batch_size": 1,
  "max_cases_per_dataset": null,
  "transform": {
    "spacing_type": "original",
    "spacing": null,
    "select_channel": 0,
    "val_patch_size": null,
    "divisible_k": 4,
    "random_aug": false
  },
  "infer": {
    "use_sliding_window": false,
    "roi_size": [256, 256, 128],
    "sw_batch_size": 1,
    "overlap": 0.0
  },
  "datasets": [
    {
      "name": "ct_set_a",
      "modality": "ct",
      "data_root": "/path/to/ct_set_a",
      "datalist_json": "/path/to/ct_set_a_test.json",
      "datalist_key": "testing"
    },
    {
      "name": "mr_set_b",
      "modality": "mri",
      "samples": [
        {"image": "/abs/path/case001.nii.gz"},
        {"image": "relative/to/data_root/case002.nii.gz"}
      ],
      "data_root": "/optional/root/for_relative_paths"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.utils import set_determinism

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAISI VAE reconstruction performance for 512x512x256 volumes.")
    parser.add_argument("--config", required=True, help="Path to the evaluation JSON config.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_samples(dataset_cfg: dict) -> list[dict]:
    name = dataset_cfg["name"]
    modality = dataset_cfg["modality"].lower()
    if modality not in {"ct", "mri"}:
        raise ValueError(f"Dataset '{name}' has unsupported modality: {modality}")

    if "datalist_json" in dataset_cfg:
        key = dataset_cfg.get("datalist_key", "testing")
        samples = load_decathlon_datalist(dataset_cfg["datalist_json"], is_segmentation=False, data_list_key=key, base_dir=dataset_cfg.get("data_root"))
    elif "samples" in dataset_cfg:
        samples = dataset_cfg["samples"]
    else:
        raise ValueError(f"Dataset '{name}' must define 'datalist_json' or 'samples'.")

    out: list[dict] = []
    data_root = dataset_cfg.get("data_root")
    for item in samples:
        if "image" not in item:
            raise ValueError(f"Dataset '{name}' contains item without 'image'.")
        image_path = Path(item["image"])
        if (not image_path.is_absolute()) and data_root:
            image_path = Path(data_root) / image_path
        out.append({"image": str(image_path), "class": modality, "dataset": name})
    return out


def _compute_metrics(images: torch.Tensor, reconstruction: torch.Tensor, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> dict[str, float]:
    l1 = torch.mean(torch.abs(images - reconstruction)).item()
    mse = F.mse_loss(reconstruction, images).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    kl = KL_loss(z_mu, z_sigma).item()
    return {"l1": l1, "mse": mse, "psnr": psnr, "kl": kl}


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.config)

    seed = int(cfg.get("seed", 42))
    set_determinism(seed=seed)
    torch.manual_seed(seed)

    network_cfg = _load_json(cfg["network_config"])
    for k, v in network_cfg.items():
        setattr(args, k, v)

    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg.get("amp", True))

    model = define_instance(args, "autoencoder_def")
    state_dict = torch.load(cfg["vae_checkpoint"], map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    transform_cfg = cfg.get("transform", {})
    eval_transform = VAE_Transform(
        is_train=False,
        random_aug=bool(transform_cfg.get("random_aug", False)),
        k=int(transform_cfg.get("divisible_k", 4)),
        patch_size=[64, 64, 64],
        val_patch_size=transform_cfg.get("val_patch_size"),
        spacing_type=transform_cfg.get("spacing_type", "original"),
        spacing=transform_cfg.get("spacing"),
        select_channel=int(transform_cfg.get("select_channel", 0)),
    )

    infer_cfg = cfg.get("infer", {})
    if infer_cfg.get("use_sliding_window", False):
        inferer = SlidingWindowInferer(
            roi_size=infer_cfg.get("roi_size", [256, 256, 128]),
            sw_batch_size=int(infer_cfg.get("sw_batch_size", 1)),
            overlap=float(infer_cfg.get("overlap", 0.0)),
            progress=False,
            device=torch.device("cpu"),
            sw_device=device,
        )
    else:
        inferer = SimpleInferer()

    all_rows: list[dict] = []
    max_cases = cfg.get("max_cases_per_dataset")

    for dataset_cfg in cfg["datasets"]:
        dataset_name = dataset_cfg["name"]
        samples = _prepare_samples(dataset_cfg)
        if max_cases is not None:
            samples = samples[: int(max_cases)]

        ds = CacheDataset(
            data=samples,
            transform=eval_transform,
            cache_rate=1.0,
            num_workers=int(cfg.get("num_workers", 4)),
        )
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 1)),
            num_workers=int(cfg.get("num_workers", 4)),
            shuffle=False,
        )

        for batch in loader:
            images = batch["image"].to(device).contiguous()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                if isinstance(inferer, SlidingWindowInferer):
                    reconstruction, z_mu, z_sigma = dynamic_infer(inferer, model, images)
                else:
                    reconstruction, z_mu, z_sigma = model(images)
            dt = time.perf_counter() - t0

            metrics = _compute_metrics(images, reconstruction, z_mu, z_sigma)
            voxels = float(images.shape[2] * images.shape[3] * images.shape[4])
            rows = images.shape[0]
            for i in range(rows):
                all_rows.append(
                    {
                        "dataset": dataset_name,
                        "image": batch["image_meta_dict"]["filename_or_obj"][i],
                        "l1": metrics["l1"],
                        "mse": metrics["mse"],
                        "psnr": metrics["psnr"],
                        "kl": metrics["kl"],
                        "seconds": dt / rows,
                        "mega_voxels_per_sec": (voxels / 1e6) / max(dt / rows, 1e-8),
                    }
                )

    output_dir = Path(cfg.get("output_dir", "./runs/vae_eval_512x512x256"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "case_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    by_dataset: dict[str, dict[str, float]] = {}
    for row in all_rows:
        name = row["dataset"]
        if name not in by_dataset:
            by_dataset[name] = {"count": 0, "l1": 0.0, "mse": 0.0, "psnr": 0.0, "kl": 0.0, "seconds": 0.0}
        by_dataset[name]["count"] += 1
        for key in ["l1", "mse", "psnr", "kl", "seconds"]:
            by_dataset[name][key] += float(row[key])

    summary = {"global": {"count": len(all_rows)}}
    for k in ["l1", "mse", "psnr", "kl", "seconds"]:
        summary["global"][k] = sum(float(r[k]) for r in all_rows) / len(all_rows)

    for name, acc in by_dataset.items():
        c = acc.pop("count")
        summary[name] = {"count": c, **{k: v / c for k, v in acc.items()}}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved case-level metrics to: {csv_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()
