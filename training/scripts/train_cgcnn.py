import os
import sys
import torch
import logging
from pathlib import Path

# Project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

from training.config import TrainConfig
from training.trainer import Trainer
from training.evaluator import evaluate
from training.utils import set_seed, build_dataloaders
from training.models.cgcnn import CGCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = TrainConfig.from_project_root(PROJECT_ROOT)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, scaler = build_dataloaders(config)

    # Model
    model = CGCNN(
        node_input_dim=config.node_input_dim,
        edge_input_dim=config.edge_input_dim,
        node_hidden_dim=config.node_hidden_dim,
        n_conv_layers=config.n_conv_layers,
        n_targets=config.n_targets,
    ).to(device)

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        scaler=scaler,
    )

    trainer.train(train_loader, val_loader)

    # Evaluation
    evaluate(
        model=trainer.model,
        test_loader=test_loader,
        scaler=scaler,
        device=device,
        config=config,
    )


if __name__ == "__main__":
    main()
