import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=50,
        lr=1e-3,
        save_path="best_lstm_model.pth",
        experiment=None,
        patience=4
    ):
    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_losses, val_losses = [], []
    maes, rmses, r2s = [], [], []

    log_interval = 1
    actual_epochs = 0

    for epoch in range(n_epochs):
        actual_epochs += 1
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{n_epochs}", leave=False)

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # VALIDAZIONE
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val)
                val_preds.append(y_pred.cpu().numpy())
                val_targets.append(y_val.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)  # (batch, seq_len, output_size)
        val_targets = np.concatenate(val_targets, axis=0)

        # Flatten per le metriche
        val_preds_flat = val_preds.reshape(-1, val_preds.shape[-1])
        val_targets_flat = val_targets.reshape(-1, val_targets.shape[-1])

        val_loss = mean_squared_error(val_targets_flat, val_preds_flat)
        mae = mean_absolute_error(val_targets_flat, val_preds_flat)
        rmse = np.sqrt(val_loss)
        r2 = r2_score(val_targets_flat, val_preds_flat)

        scheduler.step()

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            if experiment is not None:
                experiment.log_model("best_lstm_model", save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — no improvement for {patience} consecutive epochs.")
                if experiment:
                    experiment.log_other("early_stopping", f"Triggered at epoch {epoch+1}")
                break

        if experiment is not None and (epoch % log_interval == 0 or epoch == n_epochs - 1):
            experiment.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "lr": scheduler.optimizer.param_groups[0]['lr']
            }, step=epoch)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Log finale e grafici
    if experiment:
        experiment.log_other("best_val_loss", best_val_loss)
        experiment.log_other("best_epoch", best_epoch)
        experiment.log_other("total_epochs", actual_epochs)

        def plot_and_log(y_values, title, ylabel, metric_name):
            actual_epochs = len(y_values)
            x_values = np.arange(1, actual_epochs + 1)
            df = pd.DataFrame({'Epoch': x_values, 'Loss': y_values})

            plt.figure(figsize=(8, 4))
            sns.lineplot(data=df, x='Epoch', y='Loss', marker="o")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            experiment.log_figure(figure_name=metric_name, figure=plt)
            plt.close()

        plot_and_log(train_losses, "Train Loss over Epochs", "Loss", "Train Loss")
        plot_and_log(val_losses, "Validation Loss over Epochs", "Loss", "Validation Loss")

        # Plot finale
        n_plot = min(300, len(val_targets))
        val_preds_flat = val_preds.reshape(-1, val_preds.shape[-1])
        val_targets_flat = val_targets.reshape(-1, val_targets.shape[-1])

        plt.figure(figsize=(12, 5))
        plt.plot(val_targets_flat[:n_plot], label='True', linewidth=2)
        plt.plot(val_preds_flat[:n_plot], label='Predicted', linestyle='--')
        plt.title("Validation Predictions vs. Ground Truth")
        plt.xlabel("Sample")
        plt.ylabel("Leakage")
        plt.legend()
        plt.tight_layout()
        experiment.log_figure(figure_name="Final Prediction vs Ground Truth", figure=plt)
        plt.close()