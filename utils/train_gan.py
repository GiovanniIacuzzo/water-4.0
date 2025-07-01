import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_gan(
    generator,
    discriminator,
    train_loader,
    val_loader,
    noise_dim,
    n_epochs=100,
    lambda_mse=10.0,
    lr_G=1e-4,
    lr_D=1e-4,
    patience=5,
    save_path_G="best_generator.pth",
    save_path_D="best_discriminator.pth"
):
    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilizzato: {device}")

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    opt_G = optim.AdamW(generator.parameters(), lr=lr_G, weight_decay=1e-4)
    opt_D = optim.AdamW(discriminator.parameters(), lr=lr_D, weight_decay=1e-4)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=n_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=n_epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    history = {
        "loss_D": [],
        "loss_G": [],
        "val_mse_G": [],
    }

    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        total_loss_D = 0.0
        total_loss_G = 0.0

        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{n_epochs}", leave=False)

        for X_cond, X_target in progress_bar:
            X_cond = X_cond.to(device)
            X_target = X_target.to(device)
            batch_size = X_cond.size(0)

            ### === 1. Train Discriminator === ###
            z = torch.randn(batch_size, noise_dim).to(device)
            with torch.no_grad():
                X_fake = generator(X_cond, z)

            real_input = torch.cat([X_cond, X_target], dim=1)
            fake_input = torch.cat([X_cond, X_fake], dim=1)

            D_real = discriminator(real_input)
            D_fake = discriminator(fake_input)

            label_real = torch.ones_like(D_real)
            label_fake = torch.zeros_like(D_fake)

            loss_D_real = bce_loss(D_real, label_real)
            loss_D_fake = bce_loss(D_fake, label_fake)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            ### === 2. Train Generator === ###
            z = torch.randn(batch_size, noise_dim).to(device)
            X_fake = generator(X_cond, z)

            fake_input = torch.cat([X_cond, X_fake], dim=1)
            D_fake = discriminator(fake_input)

            loss_G_bce = bce_loss(D_fake, torch.ones_like(D_fake))
            loss_G_mse = mse_loss(X_fake, X_target)
            loss_G = loss_G_bce + lambda_mse * loss_G_mse

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()

        avg_loss_D = total_loss_D / len(train_loader)
        avg_loss_G = total_loss_G / len(train_loader)
        history["loss_D"].append(avg_loss_D)
        history["loss_G"].append(avg_loss_G)

        ### === VALIDATION MSE === ###
        generator.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                z = torch.randn(X_val.size(0), noise_dim).to(device)
                y_pred = generator(X_val, z)
                val_preds.append(y_pred.cpu().numpy())
                val_targets.append(y_val.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_mse = np.mean((val_preds - val_targets) ** 2)
        history["val_mse_G"].append(val_mse)

        print(f"Epoch {epoch+1}/{n_epochs} | loss_D: {avg_loss_D:.4f} | loss_G: {avg_loss_G:.4f} | val_MSE_G: {val_mse:.6f}")

        scheduler_D.step()
        scheduler_G.step()

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(generator.state_dict(), save_path_G)
            torch.save(discriminator.state_dict(), save_path_D)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping attivato a epoca {epoch+1} (no improvement in {patience} epochs)")
                break

    print(f"\nMiglior epoca: {best_epoch} con val MSE: {best_val_loss:.6f}")

    ### === PLOTTING === ###
    def plot_metric(values, title, ylabel, filename):
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=np.arange(1, len(values)+1), y=values)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_metric(history["loss_D"], "Discriminator Loss", "Loss", "loss_D.png")
    plot_metric(history["loss_G"], "Generator Loss", "Loss", "loss_G.png")
    plot_metric(history["val_mse_G"], "Validation MSE (Generator)", "MSE", "val_mse_G.png")

    # Plot finale: predizioni vs reali
    """ n_plot = min(300, len(val_preds))
    plt.figure(figsize=(12, 5))
    for i in range(n_plot):
        plt.plot(val_targets[i, :, 0].cpu(), label=f"Sample {i}")
    plt.plot(val_preds[:n_plot], label='Generated', linestyle='--')
    plt.title("Final Prediction vs Ground Truth")
    plt.xlabel("Sample")
    plt.ylabel("Leakage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_prediction_vs_ground_truth.png")
    plt.close() """

    # Heatmap errori assoluti
    """ errors = np.abs(val_targets - val_preds)
    plt.figure(figsize=(10, 4))
    sns.heatmap(errors.T, cmap="Reds", cbar=True)
    plt.title("Absolute Error Heatmap (Validation Set)")
    plt.xlabel("Time Step")
    plt.ylabel("Output Dimension")
    plt.tight_layout()
    plt.savefig("error_heatmap.png")
    plt.close() """

    return history
