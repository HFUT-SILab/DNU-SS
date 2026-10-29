"""Extract DNU-SS descriptors, evaluate every unordered pair, save redrawable ROC."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import load_checkpoint
from .data import read_split, PalmprintROIs
from .metrics import pair_scores, verification_metrics
from .runtime import configure
from .resources import DATASETS, split_file
from .ablations import validate_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DNU-SS verification descriptors"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-file", type=Path, help="Override the packaged test split"
    )
    parser.add_argument(
        "--dataset", choices=DATASETS, help="Defaults to checkpoint metadata"
    )
    parser.add_argument(
        "--protocol", choices=["open", "closed"], help="Defaults to checkpoint metadata"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lif-backend", choices=["auto", "torch", "cupy"], default="auto"
    )
    parser.add_argument("--save-pair-scores", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive; workers must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a nonempty evaluation: {args.output}"
        )
    _, device = configure(0, args.device)
    model, payload = load_checkpoint(
        args.checkpoint, device, "torch" if device.type == "cpu" else args.lif_backend
    )
    model.eval()
    expected_counts = validate_config(model.config)
    dataset = args.dataset or payload["training"].get("dataset")
    protocol = args.protocol or payload["training"].get("protocol")
    if args.test_file is not None:
        records = read_split(args.test_file, args.data_root)
    else:
        if dataset is None or protocol is None:
            parser.error(
                "Provide --test-file or dataset/protocol metadata for the packaged split"
            )
        with split_file(dataset, protocol, "test") as path:
            records = read_split(path, args.data_root)
    loader = DataLoader(
        PalmprintROIs(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    descriptors, labels = [], []
    print(
        f"DNU-SS evaluation | {len(records)} ROI images | batch={args.batch_size}",
        flush=True,
    )
    print(f"ABLATION={model.config.ablation} EXPECTED_K={expected_counts}", flush=True)
    with torch.no_grad():
        for images, identities in loader:
            descriptors.append(model.extract_features(images.to(device)).cpu().numpy())
            labels.append(identities.numpy())
    features, labels = np.concatenate(descriptors), np.concatenate(labels)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", features)
    np.save(args.output / "labels.npy", labels)
    scores, pair_labels = pair_scores(features, labels)
    result, fpr, tpr, thresholds = verification_metrics(scores, pair_labels)
    result.update(
        model="DNU-SS",
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        feature_shape=list(features.shape),
        training=payload["training"],
        actual_update_counts=[
            block.last_selection.shape[-1] for block in model.sgsb.spgtr_blocks
        ]
        if model.sgsb is not None
        else [],
        ablation=model.config.ablation,
        model_config=model.config.to_dict(),
        strict_load=True,
        precision="FP32; backend defaults retained",
        numerical_settings=dict(
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        ),
    )
    if result["actual_update_counts"] != expected_counts:
        raise ValueError("Restored checkpoint and actual update budgets do not match")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output / "roc.npz", fpr=fpr, tpr=tpr, thresholds=thresholds
    )
    if args.save_pair_scores:
        np.save(args.output / "pair_scores.npy", scores)
        np.save(args.output / "pair_labels.npy", pair_labels)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    )
    plt.rcParams["pdf.fonttype"] = 42
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.semilogx(fpr, tpr, label="DNU-SS")
    axis.set(xlabel="FAR", ylabel="TAR", xlim=(1e-6, 1), ylim=(0, 1))
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(args.output / "roc.png", dpi=300)
    figure.savefig(args.output / "roc.pdf")
    plt.close(figure)
    for name in ("AUC", "EER", "d_prime"):
        print(f"{name}: {result[name]:.4f}", flush=True)
    print(f"DNU-SS evaluation complete | results={args.output}", flush=True)


if __name__ == "__main__":
    main()

