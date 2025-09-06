from pathlib import Path
from typing import Iterable, Tuple
import cv2

from .mlf_parser import parse_mlf_for_record
from ..phonemes import PHONEME_TO_VISEME


def extract_frames_from_video(
    input_video: Path,
    segments: Iterable[Tuple[float, float, str]],
    out_dir: Path,
    frame_size: Tuple[int, int] = (64, 64),
    fps: float = 25.0
):
    """
    Extract viseme-aligned frames from a video.

    Args:
        input_video (Path): Path to input video
        segments (Iterable[Tuple[float, float, str]]): list of (start_s, end_s, phoneme)
        out_dir (Path): Root directory to save frames (viseme subfolders will be created)
        frame_size (Tuple[int,int]): Resize frames to this size (default 64x64)
        fps (float): Number of frames per second to extract
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {input_video}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    for i, (start_s, end_s, phon) in enumerate(segments):
        viseme = PHONEME_TO_VISEME.get(phon)
        if not viseme or end_s <= start_s:
            continue

        vis_dir = out_dir / viseme / input_video.stem
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Compute frame indices
        start_frame = int(start_s * video_fps)
        end_frame = int(end_s * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for f_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            out_file = vis_dir / f"{i:04d}_{f_idx - start_frame:02d}_{phon}.png"
            cv2.imwrite(str(out_file), frame)

    cap.release()


def extract_all_frames_from_dir(
    raw_videos: Path,
    mlf_path: Path,
    viseme_out: Path,
    record_prefix: str = "",
    frame_size: Tuple[int,int] = (64, 64),
    fps: float = 25.0
):
    """
    Extract frames for all videos in a directory.
    """
    viseme_out.mkdir(parents=True, exist_ok=True)

    for mp4 in raw_videos.glob("*.mp4"):
        target_rec = record_prefix + f"{mp4.stem}.rec"
        segs = parse_mlf_for_record(mlf_path, target_rec)
        if not segs:
            print(f"⚠️ No segments found for {mp4.name}")
            continue

        extract_frames_from_video(mp4, segs, viseme_out, frame_size, fps)


# Example Usage
# extract_all_frames_from_dir(
#     raw_videos=Path("data/raw/speaker01"),
#     mlf_path=Path("alignments/all.mlf"),
#     viseme_out=Path("data/viseme_clips"),
#     frame_size=(64,64),
#     fps=25.0
# )
