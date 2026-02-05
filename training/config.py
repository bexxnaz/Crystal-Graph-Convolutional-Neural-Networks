from dataclasses import dataclass
from pathlib import Path


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
    def from_project_root(cls, root: Path):
        return cls(
            data_path=root / "data/materials_project/all_data.json",
            save_dir=root / "models",
            output_dir=root / "docs/screenshots",
        )
