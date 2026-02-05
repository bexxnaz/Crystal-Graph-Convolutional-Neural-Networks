import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device, scaler):
        self.model = model
        self.config = config
        self.device = device
        self.scaler = scaler

        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10
        )
        self.criterion = nn.L1Loss()

        self.best_val = float("inf")
        self.early_stop = 0

        config.save_dir.mkdir(exist_ok=True)

    def train(self, train_loader, val_loader):
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader)
            val_mae = self._validate(val_loader)

            self.scheduler.step(val_mae)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f}"
                )

            if val_mae < self.best_val:
                self.best_val = val_mae
                self.early_stop = 0
                self._save_checkpoint()
            else:
                self.early_stop += 1
                if self.early_stop >= self.config.patience:
                    logger.info("Early stopping triggered")
                    break

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            targets = batch["target"].to(self.device)
            targets_n = self.scaler.normalize(targets)

            preds_n = self.model(
                batch["atom_fea"].to(self.device),
                batch["nbr_fea"].to(self.device),
                batch["nbr_idx"].to(self.device),
                batch["batch"].to(self.device),
            )

            loss = self.criterion(preds_n, targets_n)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)

        return total_loss / len(loader.dataset)

    def _validate(self, loader):
        self.model.eval()
        mae_sum = 0.0
        n = 0

        with torch.no_grad():
            for batch in loader:
                targets = batch["target"].to(self.device)
                preds_n = self.model(
                    batch["atom_fea"].to(self.device),
                    batch["nbr_fea"].to(self.device),
                    batch["nbr_idx"].to(self.device),
                    batch["batch"].to(self.device),
                )
                preds = self.scaler.denormalize(preds_n)
                mae_sum += torch.sum(torch.abs(preds - targets)).item()
                n += targets.numel()

        return mae_sum / n

    def _validate(self, loader):
        self.model.eval()
        total_mae = torch.zeros(self.config.n_targets, device=self.device)
        total_count = 0

        with torch.no_grad():
            for batch in loader:
                targets = batch["target"].to(self.device)
                preds_n = self.model(
                    batch["atom_fea"].to(self.device),
                    batch["nbr_fea"].to(self.device),
                    batch["nbr_idx"].to(self.device),
                    batch["batch"].to(self.device),
                )
                preds = self.scaler.denormalize(preds_n)
                total_mae += torch.sum(torch.abs(preds - targets), dim=0)
                total_count += targets.size(0)

        return (total_mae / total_count).mean().item()

    def _save_checkpoint(self):
        path = self.config.save_dir / "cgcnn_best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler.mean,
                "scaler_std": self.scaler.std,
                "config": self.config.__dict__,
            },
            path,
        )
