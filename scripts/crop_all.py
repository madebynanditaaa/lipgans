#!/usr/bin/env python3
"""
Crop mouth ROI from per-viseme clips and save 64x64 PNG frames.

Usage:
  python scripts/crop_all.py --paths config/paths.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lipgans.config import Config
from lipgans.data.crop_mouth import crop_all_frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to config/paths.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.paths)
    print(f"[INFO] Viseme clips dir: {cfg.paths.viseme_clips_dir}")
    print(f"[INFO] Cropped output dir: {cfg.paths.cropped_dir}")

    crop_all_frames(cfg.paths.viseme_clips_dir, cfg.paths.cropped_dir, resize=tuple(cfg.train.img_size))
    print("[DONE] Cropping finished.")

if __name__ == "__main__":
    main()
