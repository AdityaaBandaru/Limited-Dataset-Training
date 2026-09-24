"""Re-evaluate a saved run and redo validation-based threshold optimization.

Usage:
    python -m src.eval.evaluate --run-dir outputs/runs/<run_name>
    python -m src.eval.evaluate --run-dir outputs/runs/<run_name> --val-max-samples-per-class 0  # full val set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.chest_xray import ChestXrayDataset, build_transforms
from src.eval.metrics import best_threshold, compute_metrics, save_plots, save_sweep_csv, threshold_sweep
from src.models.efficientnet import build_model
from src.train.train_classifier import predict
from src.utils.common import get_device, save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None, help="Defaults to the run's data dir.")
    p.add_argument("--val-max-samples-per-class", type=int, default=None,
                   help="Defaults to the run's setting; 0 = use every validation image.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.run_dir / "model.pt", map_location="cpu")
    cfg = ckpt["config"]

    data_dir = args.data_dir or Path(cfg["data_dir"])
    n_val = cfg["val_max_samples_per_class"] if args.val_max_samples_per_class is None else args.val_max_samples_per_class
    val_ds = ChestXrayDataset(data_dir / "val", n_val or None, cfg["seed"],
                              build_transforms(cfg["image_size"], train=False))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Val: {len(val_ds)} images {val_ds.class_counts()}")

    model = build_model(pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    _, y_true, y_prob = predict(model, loader, nn.BCEWithLogitsLoss(), device)
    at_default = compute_metrics(y_true, y_prob, 0.5)
    sweep = threshold_sweep(y_true, y_prob)
    best = best_threshold(sweep, "f1")

    out_dir = args.run_dir / f"eval_val{n_val or 'all'}"
    out_dir.mkdir(exist_ok=True)
    save_json({"threshold_0.50": at_default, "best_f1_threshold": best}, out_dir / "metrics.json")
    save_sweep_csv(sweep, out_dir / "threshold_sweep.csv")
    save_plots(y_true, y_prob, sweep, best, out_dir)

    for name, m in (("threshold 0.50", at_default), (f"best t={best['threshold']:.2f}", best)):
        cm = m["confusion_matrix"]
        print(f"{name:15s}: acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f} [TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}]")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
