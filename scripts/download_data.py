"""Download the Kaggle chest X-ray dataset via kagglehub and lay it out as

    data/chest_xray/train/{NORMAL,PNEUMONIA}
    data/chest_xray/val/{NORMAL,PNEUMONIA}

Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The original dataset ships train/val/test splits, but its val split only has
8 images per class -- too small for a balanced 50-per-class validation set.
We therefore build our `val/` folder from the original val + test splits
(both disjoint from the original train split). Balanced subsampling to
N per class happens at training time, not here.

Usage:
    python scripts/download_data.py [--out data/chest_xray] [--symlink] [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
CLASSES = ("NORMAL", "PNEUMONIA")
IMG_EXTS = {".jpeg", ".jpg", ".png"}

# Our split -> original splits it is built from.
SPLIT_SOURCES = {
    "train": ("train",),
    "val": ("val", "test"),
}


def find_dataset_root(download_dir: Path) -> Path:
    """Locate the folder containing train/ val/ test/ (skipping __MACOSX copies)."""
    candidates = []
    for dirpath, dirnames, _ in os.walk(download_dir):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]
        p = Path(dirpath)
        if all((p / s / c).is_dir() for s in ("train", "val", "test") for c in CLASSES):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Could not find train/val/test folders under {download_dir}")
    # Prefer the shallowest match.
    return min(candidates, key=lambda p: len(p.parts))


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def build_split(src_root: Path, out_root: Path, split: str, symlink: bool) -> dict[str, int]:
    counts = {}
    for cls in CLASSES:
        dst_dir = out_root / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for orig_split in SPLIT_SOURCES[split]:
            for img in list_images(src_root / orig_split / cls):
                # Prefix with the original split to avoid filename collisions.
                dst = dst_dir / f"{orig_split}_{img.name}"
                if dst.exists() or dst.is_symlink():
                    n += 1
                    continue
                if symlink:
                    dst.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dst)
                n += 1
        counts[cls] = n
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/chest_xray"), help="Output directory.")
    parser.add_argument("--symlink", action="store_true", help="Symlink into the kagglehub cache instead of copying.")
    parser.add_argument("--force", action="store_true", help="Delete the output directory first.")
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"Downloading {KAGGLE_DATASET} via kagglehub (cached after first run)...")
    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src_root = find_dataset_root(download_dir)
    print(f"Dataset found at: {src_root}")

    if args.force and args.out.exists():
        shutil.rmtree(args.out)

    for split in SPLIT_SOURCES:
        counts = build_split(src_root, args.out, split, args.symlink)
        sources = " + ".join(SPLIT_SOURCES[split])
        print(f"  {split:5s} (from {sources}): " + ", ".join(f"{c}={n}" for c, n in counts.items()))

    print(f"Done. Data written to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
