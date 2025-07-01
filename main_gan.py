import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import torch
from torch.utils.data import DataLoader, random_split

from utils.dataset_gan import GANLeakageDataset
from models.generatore import GeneratorCGAN
from models.discriminatore import DiscriminatorCGAN
from utils.train_gan import train_gan
from utils.generate_scenario import generate_scenario
import numpy as np

if __name__ == "__main__":
    # ----------------------------
    # CONFIG
    # ----------------------------
    batch_size = 64
    seq_len = 288
    forecast_horizon = 48
    noise_dim = 64
    n_epochs = 1
    lr = 1e-4
    lambda_reg = 10.0
    patience = 5

    # ----------------------------
    # DATASET
    # ----------------------------
    dataset = GANLeakageDataset(
        demand_files=["data/2018_SCADA_Demands.csv"],
        flow_files=["data/2018_SCADA_Flows.csv"],
        level_files=["data/2018_SCADA_Levels.csv"],
        pressure_files=["data/2018_SCADA_Pressures.csv"],
        seq_len=seq_len,
        forecast_horizon=forecast_horizon
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = dataset[0][0].shape[-1]  # n_features

    # ----------------------------
    # MODELLI
    # ----------------------------
    G = GeneratorCGAN(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        forecast_horizon=forecast_horizon,
        noise_dim=noise_dim,
        dropout=0.3
    )

    D = DiscriminatorCGAN(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        seq_len_total=seq_len + forecast_horizon,
        dropout=0.3
    )

    # ----------------------------
    # TRAINING
    # ----------------------------
    train_gan(
        generator=G,
        discriminator=D,
        train_loader=train_loader,
        val_loader=val_loader,
        noise_dim=noise_dim,
        n_epochs=n_epochs,
        lambda_mse=lambda_reg,
        lr_G=lr,
        lr_D=lr,
        patience=patience,
        save_path_G="best_generator.pth",
        save_path_D="best_discriminator.pth"
    )

    # ----------------------------
    # GENERAZIONE SCENARI
    # ----------------------------
    G.load_state_dict(torch.load("best_generator.pth"))

    # Prepara i dati di input condizionante per la generazione
    X_cond = []
    for x_cond_batch, _ in val_loader:
        X_cond.append(x_cond_batch)
    X_cond = torch.cat(X_cond, dim=0).numpy()

    # Genera scenari
    scenarios = generate_scenario(
        generator=G,
        X_cond=X_cond,
        noise_dim=noise_dim,
        n_scenarios=10,
        batch_size=batch_size,
        save_path="scenarios.npy",
        plot_example=True
    )

    print(f"Scenari generati: {scenarios.shape}")  # (n_samples, 10, forecast_horizon, n_features)
