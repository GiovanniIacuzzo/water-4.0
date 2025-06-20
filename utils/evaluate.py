import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

def plot_predictions_vs_truth(test_targets, test_preds, n_plot=300, title="Predictions vs Ground Truth", ylabel="Leakage", experiment=None):
    plt.figure(figsize=(12, 5))
    plt.plot(test_targets[:n_plot], label="True", linewidth=2)
    plt.plot(test_preds[:n_plot], label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if experiment:
        experiment.log_figure(figure_name=f"{title}_Figure", figure=plt)

    plt.show()

def plot_error_heatmap(test_targets, test_preds, title="Absolute Error Heatmap", experiment=None):
    errors = np.abs(test_targets - test_preds)
    plt.figure(figsize=(10, 4))
    sns.heatmap(errors.T, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Output Dimension" if errors.shape[1] > 1 else "Leakage")
    plt.tight_layout()

    if experiment:
        experiment.log_figure(figure_name=f"{title}_Figure", figure=plt)

    plt.show()

def evaluate_model(model, test_loader, experiment=None, name_prefix="Test", n_plot=300):
    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilizzato: {device}")
    
    model = model.to(device)
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred = model(x_test)
            test_preds.append(y_pred.cpu().numpy())
            test_targets.append(y_test.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)

    # Metriche
    mae = mean_absolute_error(test_targets, test_preds)
    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)

    print(f"{name_prefix} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    if experiment:
        experiment.log_metric(f"{name_prefix}_MAE", mae)
        experiment.log_metric(f"{name_prefix}_RMSE", rmse)
        experiment.log_metric(f"{name_prefix}_R2", r2)

        # Log predizioni vs target
        plot_predictions_vs_truth(test_targets, test_preds, n_plot=n_plot, title=f"{name_prefix} Predictions vs Ground Truth", experiment=experiment)

        # Log Heatmap errore assoluto
        plot_error_heatmap(test_targets, test_preds, title=f"{name_prefix} Absolute Error Heatmap", experiment=experiment)

    return mae, rmse, r2
