#!/usr/bin/env python3
"""
Create a preview grid of cropped mouth frames for quick inspection.

Usage:
  python scripts/preview_crops.py --paths config/paths.yaml --samples 5
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lipgans.config import Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to config/paths.yaml")
    ap.add_argument("--samples", type=int, default=5, help="Number of images per viseme class to show")
    args = ap.parse_args()

    cfg = Config.load(args.paths)
    cropped_root = Path(cfg.paths.cropped_dir)
    classes = sorted([p for p in cropped_root.iterdir() if p.is_dir()])

    if not classes:
        print(f"[ERROR] No classes found in {cropped_root}")
        return

    n = len(classes)
    cols = args.samples
    plt.figure(figsize=(cols * 2, n * 2))

    for i, cls in enumerate(classes):
        shown = 0
        # iterate clips in class
        for clip in sorted([p for p in cls.iterdir() if p.is_dir()]):
            for frame in sorted(clip.glob("*.png")):
                img = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
                ax = plt.subplot(n, cols, i * cols + shown + 1)
                ax.imshow(img); ax.axis('off')
                if shown == 0:
                    ax.set_title(cls.name, fontsize=9)
                shown += 1
                if shown >= cols:
                    break
            if shown >= cols:
                break

    out = Path("outputs/cropped_preview_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[OK] Preview saved to: {out}")

if __name__ == "__main__":
    main()
