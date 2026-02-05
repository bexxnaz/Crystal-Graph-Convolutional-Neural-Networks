import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

TARGETS = ["Formation Energy", "Band Gap", "Density"]
UNITS = ["eV/atom", "eV", "g/cm^3"]



def plot_scatter(y_true, y_pred, target_name, unit, save_path):
    """Generate True vs Predicted Scatter Plot."""
    plt.figure(figsize=(8, 6))
    
    # Calculate limits for identity line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * 0.05
    
    plt.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'k--', alpha=0.5, label='Ideal')
    plt.scatter(y_true, y_pred, alpha=0.6, c='royalblue', edgecolors='w', s=50)
    
    plt.xlabel(f"True {target_name} ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted {target_name} ({unit})", fontsize=12, fontweight='bold')
    plt.title(f"{target_name}: True vs Predicted", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Add metrics annotation
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_text = f"MAE: {mae:.3f}\n$R^2$: {r2:.3f}"
    plt.annotate(metrics_text, xy=(0.05, 0.9), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_dist(y_true, y_pred, target_name, unit, save_path):
    """Generate Error Distribution Histogram."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color='crimson', bins=30, alpha=0.6)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel(f"Prediction Error ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title(f"{target_name}: Error Distribution", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



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
        rmse = mean_squared_error(y_t, y_p)
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
