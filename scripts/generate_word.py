#!/usr/bin/env python3
"""
Generate a GIF and MP4 for a single word using trained viseme GANs.

Usage:
  python scripts/generate_word.py --paths config/paths.yaml --word hello
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lipgans.config import Config
from lipgans.generate.merge_gans import generate_word

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to config/paths.yaml")
    ap.add_argument("--word", required=True, help="Word to generate")
    ap.add_argument("--out", default=None, help="Optional output directory (overrides config)")
    args = ap.parse_args()

    cfg = Config.load(args.paths)
    out_dir = Path(args.out) if args.out else None

    try:
        gif_path, mp4_path = generate_word(args.word, cfg, out_dir=out_dir)
        print(f"[OK] GIF saved to: {gif_path}")
        print(f"[OK] MP4 saved to: {mp4_path}")
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")

if __name__ == "__main__":
    main()
