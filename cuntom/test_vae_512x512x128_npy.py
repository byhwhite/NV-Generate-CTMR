#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from cuntom.npy_vae_data import NpyVaeDataset, load_paths
from scripts.utils import KL_loss, define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 512x512x128 VAE on npy data.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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

    net_cfg = _load_json(cfg["network_config"])
    for k, v in net_cfg.items():
        setattr(args, k, v)

    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg.get("amp", True))

    model = define_instance(args, "autoencoder_def")
    model.load_state_dict(torch.load(cfg["vae_checkpoint"], map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    data_cfg = cfg["data"]
    paths = load_paths(data_cfg["paths_json"], key=data_cfg.get("key"), data_root=data_cfg.get("data_root"))
    if cfg.get("max_cases"):
        paths = paths[: int(cfg["max_cases"])]

    ds = NpyVaeDataset(
        paths,
        target_hw=int(cfg.get("target_hw", 512)),
        target_depth=int(cfg.get("target_depth", 128)),
        resize_if_smaller=bool(cfg.get("resize_if_smaller", True)),
    )
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 1)), num_workers=int(cfg.get("num_workers", 4)), shuffle=False)

    rows: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device)
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
            reconstruction, z_mu, z_sigma = model(images)
        dt = time.perf_counter() - t0

        metrics = _compute_metrics(images, reconstruction, z_mu, z_sigma)
        bsz = images.shape[0]
        voxels = float(images.shape[2] * images.shape[3] * images.shape[4])
        for i in range(bsz):
            rows.append(
                {
                    "image": batch["path"][i],
                    "l1": metrics["l1"],
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "kl": metrics["kl"],
                    "seconds": dt / bsz,
                    "mega_voxels_per_sec": (voxels / 1e6) / max(dt / bsz, 1e-8),
                }
            )

    out_dir = Path(cfg.get("output_dir", "./runs/vae_eval_512x512x128_npy"))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "case_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"count": len(rows)}
    for k in ["l1", "mse", "psnr", "kl", "seconds", "mega_voxels_per_sec"]:
        summary[k] = sum(float(r[k]) for r in rows) / len(rows)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
