from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NpyVaeDataset(Dataset):
    """Load .npy volumes with shape (1, C, H, W) and normalize to model-ready (1, 128, 512, 512)."""

    def __init__(
        self,
        paths: list[str],
        target_hw: int = 512,
        target_depth: int = 128,
        resize_if_smaller: bool = True,
        crop_if_larger_depth: bool = True,
    ) -> None:
        self.paths = paths
        self.target_hw = target_hw
        self.target_depth = target_depth
        self.resize_if_smaller = resize_if_smaller
        self.crop_if_larger_depth = crop_if_larger_depth

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array at {path}, got shape={arr.shape}")
        if arr.shape[0] != 1:
            raise ValueError(f"Expected first dimension=1 at {path}, got shape={arr.shape}")

        x = torch.from_numpy(arr).float().clamp(0.0, 1.0)  # [1, C, H, W]
        _, d, h, w = x.shape

        if self.resize_if_smaller and (h < self.target_hw or w < self.target_hw):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(d, self.target_hw, self.target_hw),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        if d < self.target_depth:
            pad_before = (self.target_depth - d) // 2
            pad_after = self.target_depth - d - pad_before
            x = F.pad(x, (0, 0, 0, 0, pad_before, pad_after), mode="constant", value=0.0)
        elif d > self.target_depth and self.crop_if_larger_depth:
            start = (d - self.target_depth) // 2
            x = x[:, start : start + self.target_depth, :, :]

        return {"image": x.contiguous(), "path": path}


def load_paths(path_json: str | Path, key: str | None = None, data_root: str | Path | None = None) -> list[str]:
    with open(path_json, encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        paths = obj
    elif isinstance(obj, dict):
        if key is None:
            if "paths" not in obj:
                raise ValueError(f"JSON {path_json} is dict, provide key or 'paths'.")
            paths = obj["paths"]
        else:
            paths = obj[key]
    else:
        raise ValueError(f"Unsupported JSON structure in {path_json}")

    root = Path(data_root) if data_root else None
    out: list[str] = []
    for p in paths:
        pth = Path(p)
        if root is not None and not pth.is_absolute():
            pth = root / pth
        out.append(str(pth))
    return out
