"""Two-phase transfer learning of EfficientNet-B0 on a tiny, balanced chest X-ray subset.

Phase 1: frozen backbone, train the classification head.
Phase 2: unfreeze the upper backbone blocks and fine-tune at a lower LR.

Example (primary experiment):
    python -m src.train.train_classifier --max-samples-per-class 50 --val-max-samples-per-class 50 \
        --batch-size 32 --phase1-epochs 3 --phase2-epochs 12 --seed 42 --num-workers 0
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.chest_xray import ChestXrayDataset, build_transforms
from src.eval.metrics import best_threshold, compute_metrics, save_plots, save_sweep_csv, threshold_sweep
from src.models.efficientnet import (
    build_model,
    count_trainable,
    freeze_backbone,
    set_frozen_bn_eval,
    unfreeze_top_blocks,
)
from src.utils.common import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOCI-1: small-data EfficientNet-B0 classifier")
    # Data
    p.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--max-samples-per-class", type=int, default=50, help="Training images per class.")
    p.add_argument("--val-max-samples-per-class", type=int, default=50, help="Validation images per class.")
    p.add_argument("--image-size", type=int, default=224)
    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--phase1-epochs", type=int, default=3)
    p.add_argument("--phase2-epochs", type=int, default=12)
    p.add_argument("--phase1-lr", type=float, default=5e-3)
    p.add_argument("--phase2-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--unfreeze-blocks", type=int, default=4, help="Top EfficientNet blocks to fine-tune in phase 2.")
    p.add_argument("--dropout", type=float, default=0.3)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    set_frozen_bn_eval(model)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    probs, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        logits = model(x).squeeze(1)
        total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    return total / max(n, 1), np.concatenate(labels).astype(int), np.concatenate(probs)


def run_phase(name, model, epochs, lr, args, train_loader, val_loader, criterion, device, history) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    print(f"\n=== {name}: {epochs} epochs, lr={lr:g}, trainable params={count_trainable(model):,} ===")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_prob = predict(model, val_loader, criterion, device)
        scheduler.step()
        m = compute_metrics(y_true, y_prob, 0.5)
        history.append({
            "phase": name, "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": m["accuracy"], "val_f1": m["f1"], "val_auc": m["roc_auc"],
        })
        print(f"[{name}] epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}  ({time.time() - t0:.1f}s)")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    run_name = args.run_name or (
        f"{datetime.now():%Y%m%d-%H%M%S}_n{args.max_samples_per_class}_seed{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_ds = ChestXrayDataset(
        args.data_dir / "train", args.max_samples_per_class, args.seed,
        build_transforms(args.image_size, train=True),
    )
    val_ds = ChestXrayDataset(
        args.data_dir / "val", args.val_max_samples_per_class, args.seed,
        build_transforms(args.image_size, train=False),
    )
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images {train_ds.class_counts()}")
    print(f"Val:   {len(val_ds)} images {val_ds.class_counts()}")

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---- Model ----
    model = build_model(pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict] = []

    freeze_backbone(model)
    run_phase("phase1", model, args.phase1_epochs, args.phase1_lr, args,
              train_loader, val_loader, criterion, device, history)

    unfreeze_top_blocks(model, args.unfreeze_blocks)
    run_phase("phase2", model, args.phase2_epochs, args.phase2_lr, args,
              train_loader, val_loader, criterion, device, history)

    # ---- Final evaluation (final-epoch model) ----
    _, y_true, y_prob = predict(model, val_loader, criterion, device)
    at_default = compute_metrics(y_true, y_prob, 0.5)
    sweep = threshold_sweep(y_true, y_prob)
    best = best_threshold(sweep, "f1")

    results = {"threshold_0.50": at_default, "best_f1_threshold": best}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["device"] = str(device)
    config["train_counts"] = train_ds.class_counts()
    config["val_counts"] = val_ds.class_counts()

    torch.save({"model_state": model.state_dict(), "config": config}, run_dir / "model.pt")
    save_json(config, run_dir / "config.json")
    save_json(history, run_dir / "history.json")
    save_json(results, run_dir / "metrics.json")
    save_sweep_csv(sweep, run_dir / "threshold_sweep.csv")
    with open(run_dir / "val_predictions.csv", "w") as f:
        f.write("path,label,prob\n")
        for (path, _), label, prob in zip(val_ds.samples, y_true, y_prob):
            f.write(f"{path},{label},{prob:.6f}\n")
    save_plots(y_true, y_prob, sweep, best, run_dir)

    def fmt(m):
        cm = m["confusion_matrix"]
        return (f"acc={m['accuracy']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}  "
                f"[TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}]")

    print("\n=== Final validation results ===")
    print(f"threshold 0.50 : {fmt(at_default)}")
    print(f"best t={best['threshold']:.2f} : {fmt(best)}")
    print(f"\nArtifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
