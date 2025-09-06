#!/usr/bin/env python3
"""
Slice raw videos into per-viseme clips using MLF timestamps.

Usage:
  python scripts/extract_all.py --paths config/paths.yaml --record-prefix "lipspeakers/..."
"""
import argparse
import sys
from pathlib import Path

# ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lipgans.config import Config
from lipgans.data.extract_viseme_clips import extract_all_from_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to config/paths.yaml")
    ap.add_argument("--record-prefix", default="", help="Optional record prefix used in your MLF paths")
    args = ap.parse_args()

    cfg = Config.load(args.paths)
    print(f"[INFO] raw_videos_dir = {cfg.paths.raw_videos_dir}")
    print(f"[INFO] mlf_path = {cfg.paths.mlf_path}")
    print("[INFO] Starting extraction (this calls ffmpeg for each segment)...")
    extract_all_from_dir(cfg.paths.raw_videos_dir, cfg.paths.mlf_path, cfg.paths.viseme_clips_dir, record_prefix=args.record_prefix)
    print("[DONE] Extraction complete.")

if __name__ == "__main__":
    main()
