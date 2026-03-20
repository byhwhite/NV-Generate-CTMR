#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from VAE_finetune_utils.data import load_path_json
from scripts.utils import define_instance
from test_VAE_utils.eval_utils import compute_case_metrics, maybe_make_video, save_npy, tensor_from_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MAISI VAE reconstruction quality on validation datasets.")
    parser.add_argument("--dataset_spec_json", type=str, required=True, help="Dataset spec JSON (list). val_json=None entries are skipped.")
    parser.add_argument("--network_config", type=str, default="./configs/config_network_rflow.json")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to VAE checkpoint (state_dict).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP inference.")
    parser.add_argument("--sw_roi_size", type=int, nargs=3, default=[128, 384, 384], metavar=("D", "H", "W"))
    parser.add_argument("--sw_overlap", type=float, default=0.0)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--max_cases_per_val", type=int, default=0, help="0 means use all cases in each val_json.")
    parser.add_argument("--output_npy_dir", type=str, required=True, help="Directory to save reconstructed npy files.")
    parser.add_argument("--save_video", action="store_true", help="If enabled, save left(original)-right(recon) AVI per case.")
    parser.add_argument("--video_dir", type=str, default="", help="Output video directory when --save_video is enabled.")
    parser.add_argument("--video_fps", type=int, default=16)
    parser.add_argument("--log_file", type=str, required=True, help="Path to output metrics log file.")
    return parser.parse_args()


def load_network_args(network_config: str) -> Namespace:
    args_ns = Namespace()
    with open(network_config, "r", encoding="utf-8") as f:
        net_cfg = json.load(f)
    for k, v in net_cfg.items():
        setattr(args_ns, k, v)
    return args_ns


def load_specs(spec_json: str) -> list[dict]:
    with open(spec_json, "r", encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, list):
        raise ValueError("dataset_spec_json must be a list.")
    return specs


def collect_val_cases(specs: list[dict], max_cases_per_val: int) -> list[tuple[str, str]]:
    all_cases: list[tuple[str, str]] = []
    for i, spec in enumerate(specs):
        name = str(spec.get("name", f"dataset_{i}"))
        val_json = spec.get("val_json", None)
        if val_json is None:
            continue
        val_json = str(val_json)
        paths = load_path_json(val_json)
        if max_cases_per_val > 0:
            paths = paths[:max_cases_per_val]
        all_cases.extend((name, p) for p in paths)
    return all_cases


def run_infer(autoencoder: torch.nn.Module, image: torch.Tensor, amp: bool, roi_size: list[int], sw_batch_size: int, overlap: float) -> torch.Tensor:
    device_type = image.device.type

    def _predictor(x: torch.Tensor) -> torch.Tensor:
        out = autoencoder(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=amp):
        recon = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=_predictor,
            overlap=overlap,
            progress=False,
        )
    return recon


def write_log(log_file: Path, rows: list[dict[str, str | float]], mean_metrics: dict[str, float]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# Overall mean metrics\n")
        f.write(f"mean_mse={mean_metrics['mse']:.8f}\n")
        f.write(f"mean_psnr={mean_metrics['psnr']:.8f}\n")
        f.write(f"mean_ssim={mean_metrics['ssim']:.8f}\n")
        f.write("\n# Per-case metrics\n")
        f.write("dataset\tpath\tmse\tpsnr\tssim\trecon_npy\tvideo\n")
        for row in rows:
            f.write(
                f"{row['dataset']}\t{row['path']}\t{row['mse']:.8f}\t{row['psnr']:.8f}\t{row['ssim']:.8f}"
                f"\t{row['recon_npy']}\t{row['video']}\n"
            )


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    net_args = load_network_args(args.network_config)
    net_args.autoencoder_def["num_splits"] = 1
    model = define_instance(net_args, "autoencoder_def")
    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    if "unet_state_dict" in checkpoint:
        checkpoint = checkpoint["unet_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    specs = load_specs(args.dataset_spec_json)
    val_cases = collect_val_cases(specs, args.max_cases_per_val)
    if len(val_cases) == 0:
        raise RuntimeError("No validation cases found (all val_json may be None/empty).")

    output_npy_dir = Path(args.output_npy_dir)
    output_npy_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir) if args.video_dir else output_npy_dir / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []
    metric_acc = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}

    for dataset_name, case_path in val_cases:
        image = tensor_from_npy(case_path, device=device)  # [1,1,D,H,W]
        recon = run_infer(
            autoencoder=model,
            image=image,
            amp=args.amp,
            roi_size=args.sw_roi_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.sw_overlap,
        )

        ori = image.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        rec = recon.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        ori = np.clip(ori, 0.0, 1.0)
        rec = np.clip(rec, 0.0, 1.0)

        metrics = compute_case_metrics(target=ori, pred=rec, data_range=1.0)
        for k in metric_acc:
            metric_acc[k] += metrics[k]

        recon_npy_path = save_npy(output_root=output_npy_dir, source_path=case_path, restored_volume=rec[None, ...])

        video_path = ""
        if args.save_video:
            saved_video = maybe_make_video(
                source_path=case_path,
                output_video_dir=video_dir,
                original_dhw=ori,
                recon_dhw=rec,
                fps=args.video_fps,
            )
            video_path = str(saved_video) if saved_video is not None else ""

        rows.append(
            {
                "dataset": dataset_name,
                "path": case_path,
                "mse": metrics["mse"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "recon_npy": str(recon_npy_path),
                "video": video_path,
            }
        )
        print(f"[done] {case_path} mse={metrics['mse']:.6f} psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f}")

    n = float(len(rows))
    mean_metrics = {k: v / n for k, v in metric_acc.items()}
    write_log(Path(args.log_file), rows, mean_metrics)

    print(f"Total cases: {len(rows)}")
    print(f"Mean metrics: mse={mean_metrics['mse']:.8f}, psnr={mean_metrics['psnr']:.8f}, ssim={mean_metrics['ssim']:.8f}")
    print(f"Log saved to: {args.log_file}")


if __name__ == "__main__":
    main()
