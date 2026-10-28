"""Portable ROI splits and the original foreground normalization/augmentation."""

from pathlib import Path, PureWindowsPath
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def read_split(path, data_root):
    root = Path(data_root).resolve()
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            filename, label = line.strip().rsplit(maxsplit=1)
            relative = Path(filename)
            if (
                relative.is_absolute()
                or PureWindowsPath(filename).drive
                or ".." in relative.parts
            ):
                raise ValueError(
                    f"{path}:{number}: split paths must be relative to data-root"
                )
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Split image escapes data-root")
            label = int(label)
            if label < 0:
                raise ValueError(
                    f"{path}:{number}: identity labels must be nonnegative"
                )
            records.append((resolved, label))
    if not records or len({p for p, _ in records}) != len(records):
        raise ValueError("Split is empty or contains duplicate image paths")
    return records


def validate_protocol(train, test, protocol):
    if {p for p, _ in train} & {p for p, _ in test}:
        raise ValueError("Train/test image overlap")
    train_ids, test_ids = {y for _, y in train}, {y for _, y in test}
    if protocol == "open" and train_ids & test_ids:
        raise ValueError("Open-set train/test identities overlap")
    if protocol == "closed" and train_ids != test_ids:
        raise ValueError("Closed-set train/test identities must match")
    missing = [str(p) for p, _ in train + test if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing ROI images; first: {missing[0]}"
        )
    return dict(
        protocol=protocol,
        train_samples=len(train),
        test_samples=len(test),
        train_identities=len(train_ids),
        test_identities=len(test_ids),
    )


class ForegroundNormalize:
    """Sample std over pixels > 0; zero-valued background is left unchanged."""

    def __call__(self, tensor):
        if tensor.shape[0] != 1:
            raise ValueError("Expected a grayscale tensor")
        channels, height, width = tensor.shape
        flat = tensor.view(channels, height * width)
        foreground = flat > 0
        values = flat[foreground]
        if values.numel() > 1:
            mean, std = values.mean(), values.std()
            flat[foreground] = values.sub_(mean).div_(std + 1e-6)
        elif values.numel() == 1:
            flat[foreground] = 0
        return flat.view(channels, height, width)


def roi_transform(training=False):
    operations = [T.Resize(128)]
    if training:
        operations.append(
            T.RandomChoice(
                [
                    T.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),
                    T.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice(
                        [
                            T.RandomRotation(
                                10,
                                interpolation=T.InterpolationMode.BICUBIC,
                                expand=False,
                                center=(64.0, 0.0),
                            ),
                            T.RandomRotation(
                                10,
                                interpolation=T.InterpolationMode.BICUBIC,
                                expand=False,
                                center=(0.0, 64.0),
                            ),
                        ]
                    ),
                ]
            )
        )
    operations += [T.ToTensor(), ForegroundNormalize()]
    return T.Compose(operations)


class PalmprintROIs(Dataset):
    def __init__(self, records, training=False):
        self.records = records
        self.transform = roi_transform(training)
        self.label_map = (
            {label: i for i, label in enumerate(sorted({y for _, y in records}))}
            if training
            else None
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            value = self.transform(image.convert("L"))
        if value.shape != (1, 128, 128) or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite square 128x128 ROI: {path}")
        return value, self.label_map[label] if self.label_map is not None else label
