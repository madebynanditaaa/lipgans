# src/lipgans/train/train_viseme.py
from pathlib import Path
import tensorflow as tf
from ..models.gan3d import VisemeGAN
from ..data.dataset import make_dataset
from ..utils.io import save_frames_and_models
from ..config import Config

VISEME_CLASSES = [
    "01_Closed_Lips", "02_Teeth_Touching", "03_Open_Mouth", "04_Rounded_Lips",
    "05_Tongue_Behind_Teeth", "06_Retroflex", "07_Fricative_Sibilant",
    "08_Nasal", "09_Lateral", "10_Semi_Vowel", "11_Additional_Consonants", "12_Complex_Sounds"
]

def train_one_class(cfg: Config, viseme: str, max_batches: int | None = None):
    print(f"Training viseme: {viseme}")
    ds = make_dataset(cfg.paths.cropped_dir, viseme,
                      batch_size=cfg.train.batch_size,
                      target_frames=cfg.train.target_frames,
                      img_size=tuple(cfg.train.img_size))
    gan = VisemeGAN(z_dim=cfg.train.z_dim,
                    target_frames=cfg.train.target_frames,
                    img_size=tuple(cfg.train.img_size))
    g_opt = tf.keras.optimizers.Adam(cfg.train.lr)
    d_opt = tf.keras.optimizers.Adam(cfg.train.lr)

    save_dir = Path(cfg.paths.models_root) / viseme
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.train.epochs):
        g_loss = d_loss = None
        for step, real in enumerate(ds):
            g_loss, d_loss = gan.train_step(real, g_opt, d_opt)
            if max_batches and step + 1 >= max_batches:
                break

        print(f"[{viseme}] Epoch {epoch+1}/{cfg.train.epochs} | "
              f"G={float(g_loss):.4f} D={float(d_loss):.4f}")

        if (epoch + 1) % 10 == 0:
            save_frames_and_models(gan, epoch+1, save_dir, cfg)

def train_all(cfg: Config, subset=None, max_batches: int | None = None):
    classes = subset or VISEME_CLASSES
    for v in classes:
        train_one_class(cfg, v, max_batches=max_batches)



# Example runs

# Full training (all data, all epochs):
# python scripts/train_all.py --paths config/paths.yaml --epochs 100

# Debugging (just 5 batches per epoch):
# python scripts/train_all.py --paths config/paths.yaml --epochs 2 --max_batches 