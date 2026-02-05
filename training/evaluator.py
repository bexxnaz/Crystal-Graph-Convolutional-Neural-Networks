import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from training.plots import plot_scatter, plot_error_dist


TARGETS = ["Formation Energy", "Band Gap", "Density"]
UNITS = ["eV/atom", "eV", "g/cm^3"]


def evaluate(model, test_loader, scaler, device, config):
    model.eval()

    preds_all, targets_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            preds_n = model(
                batch["atom_fea"].to(device),
                batch["nbr_fea"].to(device),
                batch["nbr_idx"].to(device),
                batch["batch"].to(device),
            )
            preds = scaler.denormalize(preds_n)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch["target"].numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    config.output_dir.mkdir(exist_ok=True)

    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        y_t, y_p = targets_all[:, i], preds_all[:, i]

        mae = mean_absolute_error(y_t, y_p)
        rmse = mean_squared_error(y_t, y_p, squared=False)
        r2 = r2_score(y_t, y_p)

        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

        plot_scatter(
            y_t, y_p, name, unit,
            config.output_dir / f"cgcnn_scatter_{i}.png"
        )
        plot_error_dist(
            y_t, y_p, name, unit,
            config.output_dir / f"cgcnn_error_{i}.png"
        )
