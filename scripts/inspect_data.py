"""Print per-split class counts and basic image statistics.

Usage:
    python scripts/inspect_data.py [--data-dir data/chest_xray] [--sample 200]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.chest_xray import CLASSES, list_samples  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--sample", type=int, default=200, help="Images per split to open for size/mode stats.")
    args = p.parse_args()

    for split in ("train", "val"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            print(f"{split}: missing ({split_dir}). Run scripts/download_data.py first.")
            continue
        samples = list_samples(split_dir)
        counts = {c: sum(1 for _, y in samples if y == i) for i, c in enumerate(CLASSES)}
        total = sum(counts.values())
        print(f"\n[{split}] {total} images")
        for c, n in counts.items():
            print(f"  {c:10s} {n:5d}  ({n / total:.1%})")

        subset = random.Random(0).sample(samples, min(args.sample, len(samples)))
        widths, heights, modes = [], [], {}
        for path, _ in subset:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes[img.mode] = modes.get(img.mode, 0) + 1
        print(f"  sampled {len(subset)}: width {min(widths)}-{max(widths)} (mean {sum(widths) / len(widths):.0f}), "
              f"height {min(heights)}-{max(heights)} (mean {sum(heights) / len(heights):.0f}), modes {modes}")


if __name__ == "__main__":
    main()
