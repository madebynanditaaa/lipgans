from pathlib import Path
import cv2
import re

def frames_to_video(frames_dir: Path, out_path: Path, fps: int = 30) -> bool:
    """
    Convert a directory of PNG frames to an MP4 video.

    Args:
        frames_dir: Path containing PNG frames.
        out_path: Output video file path.
        fps: Frames per second.

    Returns:
        True if video was successfully created, False if no frames found.
    """
    imgs = sorted(frames_dir.glob("*.png"), key=lambda p: int(re.search(r'(\d+)\.png$', p.name).group(1)))
    if not imgs:
        return False

    # Read first frame to get size
    first = cv2.imread(str(imgs[0]))
    if first is None:
        return False
    h, w = first.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Write frames
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue  # skip missing/corrupt frames
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)

    out.release()
    return True
