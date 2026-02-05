import argparse
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Paths
    data_path: Path
    save_dir: Path
    output_dir: Path

    # Training
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 40
    seed: int = 42

    # Model
    node_input_dim: int = 4
    edge_input_dim: int = 41
    node_hidden_dim: int = 64
    n_conv_layers: int = 3
    n_targets: int = 3

    num_workers: int = 0

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--data_path", type=Path, required=True)
        parser.add_argument("--save_dir", type=Path, default=Path("models"))
        parser.add_argument("--output_dir", type=Path, default=Path("docs/screenshots"))

        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--patience", type=int, default=40)
        parser.add_argument("--seed", type=int, default=42)

        args = parser.parse_args()
        return cls(**vars(args))
