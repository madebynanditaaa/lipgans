#!/usr/bin/env python3
"""
Train GANs for all viseme classes (or a subset).

Usage:
  python scripts/train_all.py --paths config/paths.yaml
  python scripts/train_all.py --paths config/paths.yaml --subset 01_Closed_Lips 03_Open_Mouth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lipgans.config import Config
from lipgans.train.train_viseme import train_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to config/paths.yaml")
    ap.add_argument("--subset", nargs="*", default=None, help="Optional list of viseme classes to train")
    args = ap.parse_args()

    cfg = Config.load(args.paths)
    print("[INFO] Starting training loop.")
    train_all(cfg, subset=args.subset)
    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
