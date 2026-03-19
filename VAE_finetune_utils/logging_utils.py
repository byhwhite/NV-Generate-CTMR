import random

import numpy as np
import torch
import wandb


def extract_orthogonal_slices(volume: torch.Tensor) -> np.ndarray:
    img = volume.detach().float().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    d, h, w = img.shape
    sd, sh, sw = d // 2, h // 2, w // 2

    axial = img[sd, :, :]
    coronal = img[:, sh, :]
    sagittal = img[:, :, sw]

    panel = np.concatenate([axial, coronal, sagittal], axis=1)
    panel = np.clip(panel, 0.0, 1.0)
    return (panel * 255).astype(np.uint8)


def pick_visualization_indices(num_items: int, max_images: int, seed: int) -> list[int]:
    if num_items <= 0:
        return []
    k = min(num_items, max_images)
    rng = random.Random(seed)
    return rng.sample(list(range(num_items)), k=k)


def log_validation_visuals_to_wandb(
    originals: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    paths: list[str],
    epoch: int,
    global_step: int,
) -> None:
    images = []
    for i, (ori, rec, path) in enumerate(zip(originals, reconstructions, paths)):
        ori_panel = extract_orthogonal_slices(ori)
        rec_panel = extract_orthogonal_slices(rec)
        combined = np.concatenate([ori_panel, rec_panel], axis=0)
        images.append(wandb.Image(combined, caption=f"epoch={epoch} step={global_step} idx={i} path={path}"))
    if images:
        wandb.log({"val/recon_panels": images}, step=global_step)
