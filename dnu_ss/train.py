"""Train the complete DNU-SS model under identity-disjoint or shared-ID splits."""

import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import DNUSS
from .ablations import PRESETS, make_config, validate_config
from .checkpoint import save_checkpoint
from .data import PalmprintROIs, read_split, validate_protocol
from .runtime import configure
from .resources import split_file
from .diagnostics import save_prior_visualization
from .models.initialization import training_parameters


def learning_rate_multiplier(epoch, warmup, epochs, minimum, base):
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    minimum_factor = minimum / base
    return minimum_factor + (1.0 - minimum_factor) * cosine


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """Reference forward/count/zero-grad/backward/update ordering."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            logits = model(images, labels)
            loss = criterion(logits, labels)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite training loss")
        total_loss += value * len(labels)
        predictions = logits.detach().max(dim=1, keepdim=True)[1]
        correct += (
            predictions.eq(labels.detach().view_as(predictions)).cpu().sum().item()
        )
        count += len(labels)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    return total_loss / count, 100.0 * correct / count


def main():
    parser = argparse.ArgumentParser(
        description="Train DNU-SS: DGT-Stem + TEB + SGSB/SPGTR"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["PolyU", "Tongji", "HFUT", "Blue", "Red", "Green"],
    )
    parser.add_argument("--protocol", choices=["open", "closed"], default="open")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="External split root; defaults to the splits shipped inside dnu_ss",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--prior-warmup-epochs", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lif-backend", choices=["auto", "torch", "cupy"], default="auto"
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ablation", choices=tuple(PRESETS), default="full")
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.save_interval) < 1
        or args.num_workers < 0
    ):
        parser.error(
            "Epochs, batch size and save interval must be positive; workers must be nonnegative"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty run: {args.output}")
    seed, device = configure(args.seed, args.device)
    with split_file(args.dataset, args.protocol, "train", args.splits_dir) as path:
        train = read_split(path, args.data_root)
    with split_file(args.dataset, args.protocol, "test", args.splits_dir) as path:
        test = read_split(path, args.data_root)
    protocol = validate_protocol(train, test, args.protocol)
    config = make_config(
        args.ablation,
        num_classes=protocol["train_identities"],
        lif_backend="torch" if device.type == "cpu" else args.lif_backend,
    )
    model = DNUSS(config).to(device)
    loader = DataLoader(
        PalmprintROIs(train, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    optimizer = torch.optim.AdamW(
        training_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: learning_rate_multiplier(
            e,
            args.warmup_epochs,
            args.epochs,
            args.minimum_learning_rate,
            args.learning_rate,
        ),
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    criterion = torch.nn.CrossEntropyLoss()
    updates = [0]

    def record_update(optimizer, positional, keywords):
        updates[0] += 1

    update_hook = optimizer.register_step_post_hook(record_update)
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run.update(
        seed=seed,
        amp=use_amp,
        model_config=config.to_dict(),
        split_report=protocol,
        checkpoint_selection="highest training classification accuracy; latest epoch wins ties",
        workflow="legacy-aligned",
        initialization="reference tensor values and post-construction RNG stream",
        numerical_settings=dict(
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        ),
    )
    (args.output / "run.json").write_text(json.dumps(run, indent=2))
    print(
        f"DNU-SS training | {args.dataset} | {args.protocol}-set | seed={seed}",
        flush=True,
    )
    print(
        f"ABLATION={config.ablation} SGSB={config.use_sgsb} PRIOR={config.use_prior} TEMPORAL={config.temporal_mode}",
        flush=True,
    )
    print(f"SPGTR update counts={validate_config(config)} | AMP={use_amp}", flush=True)
    best_accuracy = 0.0
    total_optimizer_updates = 0
    train_losses, train_accuracies = [], []
    with (args.output / "epochs.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            if model.sgsb is not None and model.sgsb.use_prior:
                model.sgsb.prior_strength = min(
                    1.0, (epoch + 1) / max(1, args.prior_warmup_epochs)
                )
            started = time.perf_counter()
            before_updates = updates[0]
            loss_value, accuracy = train_epoch(
                model, loader, optimizer, criterion, scaler, device
            )
            elapsed = time.perf_counter() - started
            optimizer_updates = updates[0] - before_updates
            if (
                model.sgsb is not None
                and model.sgsb.use_prior
                and epoch % args.save_interval == 0
            ):
                try:
                    images, _ = next(iter(loader))
                    save_prior_visualization(
                        model, images.to(device), epoch, args.output / "prior_vis"
                    )
                except Exception as error:
                    print(f"Prior visualization skipped: {error}", flush=True)
            scheduler.step()
            total_optimizer_updates += optimizer_updates
            train_losses.append(loss_value)
            train_accuracies.append(accuracy)
            record = dict(
                epoch=epoch,
                loss=loss_value,
                training_accuracy=accuracy,
                seconds=elapsed,
                next_learning_rate=scheduler.get_last_lr()[0],
                optimizer_updates=optimizer_updates,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(
                    args.output / "checkpoints/best.pt",
                    model,
                    dict(
                        epoch=epoch,
                        seed=seed,
                        best_training_accuracy=best_accuracy,
                        protocol=args.protocol,
                        dataset=args.dataset,
                        ablation=args.ablation,
                    ),
                )
            if epoch >= 300 and epoch % args.save_interval == 0:
                save_checkpoint(
                    args.output / f"checkpoints/epoch_{epoch}.pt",
                    model,
                    dict(
                        epoch=epoch,
                        seed=seed,
                        dataset=args.dataset,
                        protocol=args.protocol,
                        ablation=args.ablation,
                    ),
                )
            (args.output / "train_losses.txt").write_text(
                "\n".join(map(str, train_losses)) + "\n"
            )
            (args.output / "train_accuracy.txt").write_text(
                "\n".join(map(str, train_accuracies)) + "\n"
            )
            print(
                f"epoch {epoch}/{args.epochs - 1} | loss {record['loss']:.5f} | acc {accuracy:.3f}% | {record['seconds']:.1f}s",
                flush=True,
            )
            if optimizer_updates == 0:
                print(
                    "AMP overflow skipped all updates this epoch; reference LR schedule advanced.",
                    flush=True,
                )
    update_hook.remove()
    if total_optimizer_updates == 0:
        raise FloatingPointError(
            "No optimizer update completed; retry a diagnostic run with --no-amp"
        )
    print(
        f"DNU-SS training complete | best training accuracy={best_accuracy:.3f}%",
        flush=True,
    )
    print(f"Checkpoint: {args.output / 'checkpoints/best.pt'}", flush=True)


if __name__ == "__main__":
    main()
