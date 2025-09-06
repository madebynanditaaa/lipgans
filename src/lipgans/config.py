from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Paths:
    raw_videos_dir: Path
    mlf_path: Path
    viseme_clips_dir: Path
    cropped_dir: Path
    models_root: Path
    merge_dir: Path
    fps: int


@dataclass
class TrainCfg:
    target_frames: int = 3
    img_size: tuple = (64, 64)
    z_dim: int = 100
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-4


@dataclass
class Config:
    paths: Paths
    train: TrainCfg
    grid_samples_per_class: int = 5

    @staticmethod
    def load(yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            y = yaml.safe_load(f)

        p = y["DATA"]
        o = y["OUTPUT"]
        t = y["TRAINING"]
        v = y.get("VIZ", {})

        paths = Paths(
            raw_videos_dir=Path(p["raw_videos_dir"]),
            mlf_path=Path(p["mlf_path"]),
            viseme_clips_dir=Path(o["viseme_clips_dir"]),
            cropped_dir=Path(o["cropped_dir"]),
            models_root=Path(o["models_root"]),
            merge_dir=Path(o["merge_dir"]),
            fps=int(p["fps"]),
        )

        train = TrainCfg(
            target_frames=int(t["target_frames"]),
            img_size=tuple(t["img_size"]),
            z_dim=int(t["z_dim"]),
            batch_size=int(t["batch_size"]),
            epochs=int(t["epochs"]),
            lr=float(t["lr"]),
        )

        return Config(
            paths=paths,
            train=train,
            grid_samples_per_class=int(v.get("grid_samples_per_class", 5)),
        )
