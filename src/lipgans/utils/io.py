from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from ..config import Config

def _next_frame_index(save_dir: Path) -> int:
    """Get the next frame index based on existing saved frames."""
    imgs = sorted(save_dir.glob("epoch_*_frame_*.png"))
    if not imgs:
        return 0
    idx = [int(p.stem.split("_")[-1]) for p in imgs]
    return max(idx) + 1

def save_frames_and_models(gan, epoch: int, save_dir: Path, cfg: Config, save_full_model: bool = True):
    """
    Save sample frames and model/weights for a given epoch.

    Args:
        gan: GAN object containing `gen` and `disc`.
        epoch: Current epoch number.
        save_dir: Directory to save frames and models.
        cfg: Config object with `train.z_dim` and `train.target_frames`.
        save_full_model: If True, save the full Keras models (can be large).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate a sample clip
    z = tf.random.normal([1, cfg.train.z_dim])
    clip = gan.gen(z, training=False).numpy()[0]  # shape: [T,H,W,3]

    # Save frames
    next_idx = _next_frame_index(save_dir)
    for i in range(cfg.train.target_frames):
        frame = ((clip[i] + 1.0) * 127.5).astype('uint8')
        path = save_dir / f"epoch_{epoch:03d}_frame_{next_idx + i}.png"
        plt.imsave(path.as_posix(), frame)

    # Save weights
    gan.gen.save_weights((save_dir / f"generator_epoch_{epoch}.weights.h5").as_posix())
    gan.disc.save_weights((save_dir / f"discriminator_epoch_{epoch}.weights.h5").as_posix())

    # Optionally save full models
    if save_full_model:
        gan.gen.save((save_dir / f"generator_epoch_{epoch}.model.keras").as_posix())
        gan.disc.save((save_dir / f"discriminator_epoch_{epoch}.model.keras").as_posix()
