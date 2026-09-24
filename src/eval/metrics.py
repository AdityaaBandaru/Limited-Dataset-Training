"""Binary classification metrics and validation-based threshold optimization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    return [compute_metrics(y_true, y_prob, t) for t in thresholds]


def best_threshold(sweep: list[dict], metric: str = "f1") -> dict:
    """Highest `metric`; ties broken by the threshold closest to 0.5."""
    return max(sweep, key=lambda r: (r[metric], -abs(r["threshold"] - 0.5)))


def save_sweep_csv(sweep: list[dict], path: str | Path) -> None:
    cols = ["threshold", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sweep:
            cm = r["confusion_matrix"]
            row = [r["threshold"], r["accuracy"], r["precision"], r["recall"], r["f1"],
                   cm["tn"], cm["fp"], cm["fn"], cm["tp"]]
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")


def save_plots(y_true: np.ndarray, y_prob: np.ndarray, sweep: list[dict], best: dict, out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {best['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (validation)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix at best threshold
    cm = best["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix @ t={best['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Metrics vs threshold
    ts = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("f1", "precision", "recall", "accuracy"):
        ax.plot(ts, [r[key] for r in sweep], label=key)
    ax.axvline(best["threshold"], ls="--", c="gray", lw=1)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation metrics vs threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=150)
    plt.close(fig)
