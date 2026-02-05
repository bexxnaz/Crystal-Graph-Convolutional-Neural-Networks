import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TargetScaler:
    def __init__(self):
        self.fitted = False

    def fit(self, y):
        self.mean = y.mean(0)
        self.std = y.std(0).clamp(min=1e-8)
        self.fitted = True

    def normalize(self, y):
        assert self.fitted
        return (y - self.mean.to(y.device)) / self.std.to(y.device)


def split_indices(n, seed, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    lengths = np.floor(np.array(ratios) * n).astype(int)
    lengths[-1] = n - lengths[:-1].sum()
    splits = np.cumsum(lengths[:-1])
    return np.split(idx, splits)


def build_dataloaders(config):
    # 1. Load data
    loader = MaterialDataLoader(config.data_path)
    raw_data = loader.load()

    # 2. Dataset
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)

    # 3. Split indices
    train_idx, val_idx, test_idx = split_indices(
        n=len(dataset),
        seed=config.seed,
        ratios=(0.7, 0.1, 0.2),
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 4. Fit scaler on TRAIN only

    train_targets = torch.cat(
        [
            batch["target"]
            for batch in DataLoader(
                train_set,
                batch_size=config.batch_size,
                collate_fn=collate_batch,
            )
        ]
    )

    scaler = TargetScaler()
    scaler.fit(train_targets)

    # 5. Dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
