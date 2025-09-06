from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, Tuple

def _load_clip(clip_dir: Path, target_frames: int, img_size: Tuple[int, int] = (64, 64)) -> Optional[np.ndarray]:
    """Load frames from a clip directory, resize, normalize, and pad/subsample to target_frames.
    Returns array shape (T, H, W, 3) with values in [-1, 1], or None if clip empty.
    """
    frame_paths = sorted([p for p in clip_dir.glob("*.png")])
    imgs = []
    for p in frame_paths:
        try:
            img = Image.open(p).convert("RGB").resize(img_size)
            arr = np.asarray(img).astype(np.float32)
            imgs.append(arr)
        except Exception:
            continue

    if not imgs:
        return None

    # Temporal pad / subsample to target_frames
    if len(imgs) < target_frames:
        imgs += [imgs[-1]] * (target_frames - len(imgs))
    elif len(imgs) > target_frames:
        idx = np.linspace(0, len(imgs) - 1, target_frames).astype(int)
        imgs = [imgs[i] for i in idx]

    arr = np.stack(imgs, axis=0)  # (T, H, W, 3)
    # Normalize to [-1, 1]
    arr = arr / 127.5 - 1.0
    return arr

def make_dataset(cropped_root: Path,
                 viseme_class: str,
                 batch_size: int = 16,
                 target_frames: int = 3,
                 img_size: Tuple[int, int] = (64, 64)) -> tf.data.Dataset:
    """Create a tf.data.Dataset for a single viseme class. Yields batches of shape
       (B, T, H, W, 3) dtype float32 in [-1,1].
    """
    class_dir = Path(cropped_root) / viseme_class
    if not class_dir.exists() or not class_dir.is_dir():
        raise FileNotFoundError(f"{class_dir} not found")

    clip_dirs = [p for p in class_dir.iterdir() if p.is_dir()]

    def gen():
        for clip in clip_dirs:
            clip_arr = _load_clip(clip, target_frames, img_size)
            if clip_arr is not None:
                yield clip_arr

    output_signature = tf.TensorSpec(shape=(target_frames, img_size[0], img_size[1], 3), dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
