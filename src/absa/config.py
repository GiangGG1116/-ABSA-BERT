from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class TrainConfig:
    model_name: str = os.getenv("MODEL_NAME", "bert-base-uncased")
    train_csv: str = os.getenv("TRAIN_CSV", "./data/raw/restaurants_train.csv")
    valid_csv: str = os.getenv("VALID_CSV", "./data/raw/restaurants_test.csv")
    save_dir: str = os.getenv("SAVE_DIR", "./models")
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_length: int = 128
    seed: int = int(os.getenv("SEED", 42))

    def ensure(self) -> "TrainConfig":
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        return self