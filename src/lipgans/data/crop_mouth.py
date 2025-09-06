from pathlib import Path
import cv2
import mediapipe as mp
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
MOUTH_LANDMARKS = list(set([
    61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,
    14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,
    13,82,81,42,183,78
]))


def crop_mouth_roi(img):
    """
    Crop the mouth region from an image using MediaPipe FaceMesh.
    Returns None if no face detected.
    """
    h, w, _ = img.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [int(landmarks[i].x * w) for i in MOUTH_LANDMARKS]
        ys = [int(landmarks[i].y * h) for i in MOUTH_LANDMARKS]
        x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)
        return img[y_min:y_max, x_min:x_max]


def crop_all_frames(viseme_clips_dir: Path, cropped_dir: Path, resize=(64, 64)):
    """
    Crop mouth region from all frames in all viseme MP4 clips.
    Saves frames as PNGs in a mirrored directory structure.
    """
    cropped_dir.mkdir(parents=True, exist_ok=True)

    for viseme_dir in tqdm(list(viseme_clips_dir.iterdir()), desc="Viseme classes"):
        if not viseme_dir.is_dir():
            continue

        out_viseme_dir = cropped_dir / viseme_dir.name
        out_viseme_dir.mkdir(exist_ok=True)

        for clip_path in tqdm(list(viseme_dir.iterdir()), desc=f"Clips in {viseme_dir.name}", leave=False):
            if clip_path.suffix.lower() != ".mp4":
                continue

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print(f"⚠️ Cannot open video: {clip_path}")
                continue

            out_clip_dir = out_viseme_dir / clip_path.stem
            out_clip_dir.mkdir(exist_ok=True)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                roi = crop_mouth_roi(frame)
                if roi is None:
                    frame_idx += 1
                    continue

                roi_resized = cv2.resize(roi, resize)
                out_file = out_clip_dir / f"{frame_idx:04d}.png"
                cv2.imwrite(str(out_file), roi_resized)
                frame_idx += 1

            cap.release()


# Example Usage
# crop_all_frames(
#     viseme_clips_dir=Path("data/viseme_clips"),
#     cropped_dir=Path("data/cropped_frames"),
#     resize=(64,64)
# )
