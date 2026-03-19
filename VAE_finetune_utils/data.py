import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetSpec:
    name: str
    train_json: str
    val_json: str
    samples_per_epoch: int


@dataclass
class SampleItem:
    image: str
    dataset_name: str


def _extract_path(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("image", "path", "npy", "file"):
            if key in entry:
                return entry[key]
    raise ValueError(f"Unsupported entry format in json: {entry}")


def load_path_json(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("data", "items", "images", "files"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError(f"{json_path} must contain a list (or dict with list field).")

    return [str(_extract_path(item)) for item in payload]


def load_dataset_specs(spec_json: str) -> list[DatasetSpec]:
    with open(spec_json, "r", encoding="utf-8") as f:
        raw_specs = json.load(f)

    if not isinstance(raw_specs, list):
        raise ValueError("dataset_spec_json should be a list of dataset configs.")

    specs: list[DatasetSpec] = []
    for i, spec in enumerate(raw_specs):
        try:
            specs.append(
                DatasetSpec(
                    name=str(spec["name"]),
                    train_json=str(spec["train_json"]),
                    val_json=str(spec["val_json"]),
                    samples_per_epoch=int(spec["samples_per_epoch"]),
                )
            )
        except KeyError as err:
            raise KeyError(f"Missing key {err} in dataset spec index {i}") from err
    return specs


def build_dataset_records(specs: list[DatasetSpec]) -> tuple[dict[str, list[SampleItem]], list[SampleItem]]:
    train_by_dataset: dict[str, list[SampleItem]] = {}
    val_records: list[SampleItem] = []

    for spec in specs:
        train_paths = load_path_json(spec.train_json)
        val_paths = load_path_json(spec.val_json)

        train_by_dataset[spec.name] = [SampleItem(image=p, dataset_name=spec.name) for p in train_paths]
        val_records.extend(SampleItem(image=p, dataset_name=spec.name) for p in val_paths)

    return train_by_dataset, val_records


def sample_train_records_for_epoch(train_by_dataset: dict[str, list[SampleItem]], specs: list[DatasetSpec], seed: int) -> list[SampleItem]:
    rng = random.Random(seed)
    epoch_records: list[SampleItem] = []
    for spec in specs:
        dataset_records = train_by_dataset[spec.name]
        if len(dataset_records) == 0:
            continue
        if spec.samples_per_epoch <= len(dataset_records):
            epoch_records.extend(rng.sample(dataset_records, k=spec.samples_per_epoch))
        else:
            epoch_records.extend(rng.choices(dataset_records, k=spec.samples_per_epoch))
    rng.shuffle(epoch_records)
    return epoch_records


class NpyVolumeDataset(Dataset):
    def __init__(self, records: list[SampleItem], patch_size: list[int], is_train: bool):
        self.records = records
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _ensure_tensor_shape(volume: np.ndarray) -> torch.Tensor:
        arr = np.asarray(volume)
        if arr.ndim != 4:
            raise ValueError(f"Expected npy shape (1, D, H, W), got {arr.shape}")
        if arr.shape[0] != 1:
            raise ValueError(f"Expected first channel dim = 1, got shape {arr.shape}")
        t = torch.from_numpy(arr).float()
        return t

    def _random_crop_3d(self, image: torch.Tensor) -> torch.Tensor:
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            pad_d = max(pd - d, 0)
            pad_h = max(ph - h, 0)
            pad_w = max(pw - w, 0)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
            _, d, h, w = image.shape

        if self.is_train:
            sd = random.randint(0, d - pd)
            sh = random.randint(0, h - ph)
            sw = random.randint(0, w - pw)
        else:
            sd = max((d - pd) // 2, 0)
            sh = max((h - ph) // 2, 0)
            sw = max((w - pw) // 2, 0)

        patch = image[:, sd : sd + pd, sh : sh + ph, sw : sw + pw]
        return patch

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[3])
        return image

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        image_np = np.load(item.image)
        image = self._ensure_tensor_shape(image_np)
        image = self._random_crop_3d(image)
        if self.is_train:
            image = self._augment(image)

        image = image.clamp(-1000, 1000)
        image = (image + 1000.0) / 2000.0

        return {"image": image, "dataset_name": item.dataset_name, "path": item.image}


def create_dataloader(records: list[SampleItem], patch_size: list[int], batch_size: int, is_train: bool, num_workers: int) -> DataLoader:
    ds = NpyVolumeDataset(records=records, patch_size=patch_size, is_train=is_train)
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True, drop_last=is_train)


def ensure_paths_exist(specs: list[DatasetSpec]) -> None:
    for spec in specs:
        for p in (spec.train_json, spec.val_json):
            if not Path(p).exists():
                raise FileNotFoundError(f"Cannot find file: {p}")
