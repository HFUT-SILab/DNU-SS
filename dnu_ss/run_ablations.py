"""Run named paper ablations sequentially with one shared train/evaluate pipeline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .ablations import PRESETS, make_config, validate_config
from .config import ModelConfig
from .preflight import inspect_dataset
from .resources import DATASETS


def build_commands(args, name):
    run = args.output.resolve() / name
    train = [
        sys.executable,
        "-u",
        "-m",
        "dnu_ss.train",
        "--dataset",
        args.dataset,
        "--protocol",
        args.protocol,
        "--ablation",
        name,
        "--data-root",
        str(args.data_root.resolve()),
        "--output",
        str(run),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--save-interval",
        str(getattr(args, "save_interval", 5)),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--lif-backend",
        args.lif_backend,
        "--amp" if args.amp else "--no-amp",
    ]
    test = [
        sys.executable,
        "-u",
        "-m",
        "dnu_ss.evaluate",
        "--checkpoint",
        str(run / "checkpoints/best.pt"),
        "--data-root",
        str(args.data_root.resolve()),
        "--output",
        str(run / "evaluation"),
        "--batch-size",
        str(args.test_batch_size),
        "--num-workers",
        str(getattr(args, "test_num_workers", 0)),
        "--device",
        args.device,
        "--lif-backend",
        args.lif_backend,
    ]
    if args.splits_dir is not None:
        train += ["--splits-dir", str(args.splits_dir.resolve())]
        test += [
            "--test-file",
            str(args.splits_dir.resolve() / args.protocol / f"test_{args.dataset}.txt"),
        ]
    return train, test


def collect_result(root, name, dataset, protocol, seed):
    run = root / name
    metrics = json.loads((run / "evaluation/metrics.json").read_text())
    config = ModelConfig.from_dict(metrics["model_config"])
    counts = validate_config(config)
    if (
        config.ablation != name
        or metrics["ablation"] != name
        or metrics["actual_update_counts"] != counts
    ):
        raise ValueError(f"Saved ablation/configuration mismatch: {name}")
    metadata = metrics["training"]
    if (metadata["dataset"], metadata["protocol"], metadata["seed"]) != (
        dataset,
        protocol,
        seed,
    ):
        raise ValueError(f"Saved dataset/protocol/seed mismatch: {name}")
    checksum = hashlib.sha256((run / "checkpoints/best.pt").read_bytes()).hexdigest()
    if checksum != metrics["checkpoint_sha256"]:
        raise ValueError(f"Evaluated checkpoint hash mismatch: {name}")
    return dict(
        ablation=name,
        dataset=dataset,
        protocol=protocol,
        seed=seed,
        best_epoch=metadata["epoch"],
        AUC=metrics["AUC"],
        EER=metrics["EER"],
        d_prime=metrics["d_prime"],
        actual_K="/".join(map(str, counts)),
        checkpoint_sha256=checksum,
        result_dir=str(run / "evaluation"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--protocol", choices=("open", "closed"), default="open")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--ablations", nargs="+", choices=tuple(PRESETS), default=list(PRESETS)
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--test-num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lif-backend", choices=("auto", "torch", "cupy"), default="auto"
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating files or training",
    )
    args = parser.parse_args()
    if len(set(args.ablations)) != len(args.ablations):
        parser.error("Each ablation may occur only once in a run")
    if (
        min(args.epochs, args.batch_size, args.test_batch_size, args.save_interval) < 1
        or args.num_workers < 0
        or args.test_num_workers < 0
    ):
        parser.error("Epochs/batch sizes must be positive and workers nonnegative")
    plan = [
        dict(
            ablation=name,
            actual_K=validate_config(make_config(name)),
            commands=build_commands(args, name),
        )
        for name in args.ablations
    ]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Refusing to overwrite a nonempty group output directory")
    report = inspect_dataset(
        args.dataset, args.protocol, args.data_root, args.splits_dir
    )
    if not report["valid"]:
        raise ValueError("Dataset preflight failed: " + "; ".join(report["errors"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    rows = []
    for entry in plan:
        name = entry["ablation"]
        print(
            f"===== ABLATION={name} SEED={args.seed} EXPECTED_K={entry['actual_K']} =====",
            flush=True,
        )
        for command in entry["commands"]:
            subprocess.run(command, check=True)
        rows.append(
            collect_result(args.output, name, args.dataset, args.protocol, args.seed)
        )
        with (args.output / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"ALL DONE: {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
