#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from scripts.utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MAISI autoencoder layer-output shapes and forward returns.")
    parser.add_argument("--network_config", type=str, default="./configs/config_network_rflow.json")
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--input_npy", type=str, required=True, help="Path to one npy with shape (1, D, H, W).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--return_mode",
        type=str,
        default="original",
        choices=["original", "recon_only", "recon_clamp01", "recon_and_latent_stats"],
        help="How to adapt the model forward return for quick experiments.",
    )
    parser.add_argument("--max_layers_to_print", type=int, default=0, help="0 means print all leaf-module outputs.")
    parser.add_argument("--save_json", type=str, default="", help="Optional path to save inspection result JSON.")
    return parser.parse_args()


def _load_network_args(network_config: str) -> Namespace:
    args_ns = Namespace()
    with open(network_config, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for k, v in payload.items():
        setattr(args_ns, k, v)
    return args_ns


def _ensure_input(path: str, device: torch.device) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"Expected shape (1, D, H, W), got {arr.shape} for {path}")
    x = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).to(device=device)  # [1,1,D,H,W]
    return x


def _shape_desc(x: Any) -> Any:
    if torch.is_tensor(x):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "mean": float(x.mean().item()),
            "std": float(x.std().item()),
        }
    if isinstance(x, (tuple, list)):
        return [_shape_desc(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_desc(v) for k, v in x.items()}
    return {"type": str(type(x)), "value": str(x)}


class AutoencoderReturnAdapter(nn.Module):
    """
    Wrapper for experimenting with autoencoder forward return formats,
    without editing upstream MONAI source code.
    """

    def __init__(self, model: nn.Module, return_mode: str = "original"):
        super().__init__()
        self.model = model
        self.return_mode = return_mode

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            return out

        reconstruction, z_mu, z_sigma = out[0], out[1], out[2]

        if self.return_mode == "original":
            return reconstruction, z_mu, z_sigma
        if self.return_mode == "recon_only":
            return reconstruction
        if self.return_mode == "recon_clamp01":
            return torch.clamp(reconstruction, 0.0, 1.0)
        if self.return_mode == "recon_and_latent_stats":
            return {
                "reconstruction": reconstruction,
                "latent_mu_mean": z_mu.mean(),
                "latent_mu_std": z_mu.std(),
                "latent_sigma_mean": z_sigma.mean(),
                "latent_sigma_std": z_sigma.std(),
            }
        raise ValueError(f"Unsupported return_mode={self.return_mode}")


def _register_leaf_hooks(model: nn.Module):
    layer_outputs: OrderedDict[str, Any] = OrderedDict()
    handles = []

    def _hook_fn(name: str):
        def _fn(_module, _inp, out):
            layer_outputs[name] = _shape_desc(out)

        return _fn

    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(_hook_fn(name)))

    return layer_outputs, handles


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    net_args = _load_network_args(args.network_config)
    net_args.autoencoder_def["num_splits"] = 1
    model = define_instance(net_args, "autoencoder_def")

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    if "unet_state_dict" in checkpoint:
        checkpoint = checkpoint["unet_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()

    wrapped_model = AutoencoderReturnAdapter(model, return_mode=args.return_mode).to(device).eval()
    x = _ensure_input(args.input_npy, device)

    layer_outputs, handles = _register_leaf_hooks(model)
    use_amp = bool(args.amp and device.type == "cuda")
    try:
        with torch.no_grad():
            if use_amp:
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        out = wrapped_model(x)
                except RuntimeError as err:
                    err_msg = str(err)
                    if "Input type (c10::Half) and bias type (float) should be the same" not in err_msg:
                        raise
                    print(
                        "[warn] AMP forward failed due to half/float mismatch. "
                        "Fallback to FP32 forward."
                    )
                    out = wrapped_model(x.float())
            else:
                out = wrapped_model(x)
    finally:
        for h in handles:
            h.remove()

    result = {
        "input": _shape_desc(x),
        "return_mode": args.return_mode,
        "forward_return": _shape_desc(out),
        "layer_output_count": len(layer_outputs),
        "layer_outputs": layer_outputs,
    }

    print("\n=== Forward return summary ===")
    print(json.dumps(result["forward_return"], indent=2, ensure_ascii=False))

    print("\n=== Leaf module output shapes ===")
    to_print = list(layer_outputs.items())
    if args.max_layers_to_print > 0:
        to_print = to_print[: args.max_layers_to_print]
    for name, summary in to_print:
        if isinstance(summary, dict) and "shape" in summary:
            print(f"{name}: shape={summary['shape']} dtype={summary['dtype']}")
        else:
            print(f"{name}: {summary}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved inspection json to: {out_path}")


if __name__ == "__main__":
    main()
