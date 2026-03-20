from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity


def ensure_volume_shape(arr: np.ndarray, path: str) -> np.ndarray:
    """Validate shape=(1, D, H, W) and return float32 array."""
    vol = np.asarray(arr)
    if vol.ndim != 4 or vol.shape[0] != 1:
        raise ValueError(f"Expected shape (1, D, H, W), got {vol.shape} for: {path}")
    return vol.astype(np.float32, copy=False)


def compute_case_metrics(target: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    """Compute MSE/PSNR/SSIM for one 3D case."""
    diff = pred - target
    mse = float(np.mean(np.square(diff), dtype=np.float64))
    psnr = float(10.0 * math.log10((data_range * data_range) / max(mse, 1e-12)))

    d = target.shape[0]
    ssim_values: list[float] = []
    for i in range(d):
        ssim_i = structural_similarity(
            target[i],
            pred[i],
            data_range=data_range,
            channel_axis=None,
        )
        ssim_values.append(float(ssim_i))
    ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def save_npy(output_root: Path, source_path: str, restored_volume: np.ndarray) -> Path:
    stem = Path(source_path).stem
    out_path = output_root / f"{stem}_recon.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, restored_volume.astype(np.float32, copy=False))
    return out_path


def maybe_make_video(
    source_path: str,
    output_video_dir: Path,
    original_dhw: np.ndarray,
    recon_dhw: np.ndarray,
    fps: int,
) -> Path | None:
    """
    Save AVI with each D-slice as one frame, left=original, right=recon.
    Input shapes: (D, H, W), values in [0,1].
    """
    try:
        import cv2
    except ImportError as err:  # pragma: no cover
        raise RuntimeError("--save_video is enabled, but OpenCV (cv2) is not installed.") from err

    output_video_dir.mkdir(parents=True, exist_ok=True)
    d, h, w = original_dhw.shape
    if recon_dhw.shape != (d, h, w):
        raise ValueError(f"Shape mismatch for video: original={original_dhw.shape}, recon={recon_dhw.shape}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = output_video_dir / f"{Path(source_path).name}.avi"
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w * 2, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_path}")

    try:
        for i in range(d):
            ori_u8 = np.clip(original_dhw[i], 0.0, 1.0)
            rec_u8 = np.clip(recon_dhw[i], 0.0, 1.0)
            ori_u8 = (ori_u8 * 255.0).round().astype(np.uint8)
            rec_u8 = (rec_u8 * 255.0).round().astype(np.uint8)

            ori_rgb = np.repeat(ori_u8[..., None], 3, axis=2)
            rec_rgb = np.repeat(rec_u8[..., None], 3, axis=2)
            frame_rgb = np.concatenate([ori_rgb, rec_rgb], axis=1)
            frame_bgr = frame_rgb[..., ::-1]
            writer.write(frame_bgr)
    finally:
        writer.release()

    return out_path


def tensor_from_npy(path: str, device: torch.device) -> torch.Tensor:
    arr = np.load(path)
    arr = ensure_volume_shape(arr, path)
    ten = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)  # [B=1,C=1,D,H,W]
    return ten
