import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_predictions_vs_truth(test_targets, test_preds, n_plot=300, title="Predictions vs Ground Truth", ylabel="Leakage", experiment=None):
    test_preds = np.squeeze(test_preds)
    test_targets = np.squeeze(test_targets)

    if test_preds.ndim > 1 and test_preds.shape[-1] > 1:
        test_preds = test_preds[:, 0]  # mostra solo la prima variabile
        test_targets = test_targets[:, 0]

    plt.figure(figsize=(12, 5), constrained_layout=True)
    plt.plot(test_targets[:n_plot], label="True", linewidth=2)
    plt.plot(test_preds[:n_plot], label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.legend()
    if experiment:
        experiment.log_figure(figure_name=f"{title}_Figure", figure=plt)
    plt.close()

def plot_error_heatmap(test_targets, test_preds, title="Absolute Error Heatmap", experiment=None):
    # Reshape sicuro
    test_targets = np.squeeze(test_targets)
    test_preds = np.squeeze(test_preds)

    if test_targets.ndim == 1:
        test_targets = test_targets[:, np.newaxis]
    if test_preds.ndim == 1:
        test_preds = test_preds[:, np.newaxis]

    # Calcolo errore
    errors = np.abs(test_targets - test_preds)

    # Controlla 2D
    if errors.ndim != 2:
        raise ValueError(f"Expected 2D error array for heatmap, got shape {errors.shape}")

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

    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # Flatten per metriche
    test_preds_flat = np.squeeze(test_preds)
    test_targets_flat = np.squeeze(test_targets)

    if test_preds_flat.ndim == 1:
        test_preds_flat = test_preds_flat[:, np.newaxis]
    if test_targets_flat.ndim == 1:
        test_targets_flat = test_targets_flat[:, np.newaxis]

    # Metriche
    mae = mean_absolute_error(test_targets_flat, test_preds_flat)
    rmse = np.sqrt(mean_squared_error(test_targets_flat, test_preds_flat))
    r2 = r2_score(test_targets_flat, test_preds_flat)

    print(f"{name_prefix} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    if experiment:
        experiment.log_metric(f"{name_prefix}_MAE", mae)
        experiment.log_metric(f"{name_prefix}_RMSE", rmse)
        experiment.log_metric(f"{name_prefix}_R2", r2)

        plot_predictions_vs_truth(test_targets, test_preds, n_plot=n_plot, title=f"{name_prefix} Predictions vs Ground Truth", experiment=experiment)
        plot_error_heatmap(test_targets, test_preds, title=f"{name_prefix} Absolute Error Heatmap", experiment=experiment)

    return mae, rmse, r2
