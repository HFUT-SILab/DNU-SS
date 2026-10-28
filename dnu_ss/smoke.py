"""Small synthetic CPU/GPU pipeline example; not a biometric performance test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .preflight import inspect_dataset
from .ablations import PRESETS, make_config, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablation", choices=tuple(PRESETS), default="full")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Refusing to overwrite a nonempty smoke directory")
    data, splits = root / "synthetic_data", root / "synthetic_splits/open"
    data.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for phase, labels in (("train", (0, 1)), ("test", (2, 3))):
        rows = []
        for label in labels:
            for sample in range(2):
                name = f"{phase}_{label}_{sample}.png"
                Image.fromarray(rng.integers(0, 256, (128, 128), dtype=np.uint8)).save(
                    data / name
                )
                rows.append(f"{name} {label}\n")
        (splits / f"{phase}_PolyU.txt").write_text("".join(rows), encoding="utf-8")
    # PolyU is only the CLI fixture name; these are explicitly synthetic inputs.
    marker = dict(
        synthetic_smoke_only=True,
        benchmark_result=False,
        seed=42,
        ablation=args.ablation,
        description="Four generated training images and four generated test images; not PolyU data.",
    )
    (root / "SMOKE_ONLY.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )
    report = inspect_dataset("PolyU", "open", data, splits.parent)
    assert report["valid"], report
    print(
        "SYNTHETIC SMOKE ONLY: outputs are not recognition benchmark results.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "dnu_ss.train",
            "--dataset",
            "PolyU",
            "--ablation",
            args.ablation,
            "--data-root",
            str(data),
            "--splits-dir",
            str(splits.parent),
            "--output",
            str(root / "run"),
            "--epochs",
            "1",
            "--seed",
            "42",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            args.device,
            "--no-amp",
            "--lif-backend",
            "torch",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "dnu_ss.evaluate",
            "--checkpoint",
            str(root / "run/checkpoints/best.pt"),
            "--test-file",
            str(splits / "test_PolyU.txt"),
            "--data-root",
            str(data),
            "--output",
            str(root / "evaluation"),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            args.device,
            "--lif-backend",
            "torch",
        ],
        check=True,
    )
    metrics_path = root / "evaluation/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(synthetic_smoke_only=True, benchmark_result=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history = [
        json.loads(line)
        for line in (root / "run/epochs.jsonl").read_text().splitlines()
    ]
    assert history[0]["optimizer_updates"] == 2
    assert metrics["feature_shape"] == [4, 1024]
    assert metrics["ablation"] == args.ablation
    assert metrics["actual_update_counts"] == validate_config(
        make_config(args.ablation)
    )
    assert metrics["num_genuine"] == 2 and metrics["num_impostor"] == 4
    for name in (
        "features.npy",
        "labels.npy",
        "metrics.json",
        "roc.npz",
        "roc.png",
        "roc.pdf",
    ):
        assert (root / "evaluation" / name).stat().st_size > 0
    print(
        "PASS: training updates, strict checkpoint load, features, metrics and ROC outputs."
    )


if __name__ == "__main__":
    main()
