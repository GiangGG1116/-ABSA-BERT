from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class TrainConfig:
    model_name: str = os.getenv("MODEL_NAME", "bert-base-uncased")
    train_csv: str = os.getenv("TRAIN_CSV", "./data/restaurants_train.csv")
    valid_csv: str = os.getenv("VALID_CSV", "./data/restaurants_test.csv")
    save_dir: str = os.getenv("SAVE_DIR", "./models")
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    seed: int = int(os.getenv("SEED", 42))

    def ensure(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        return self