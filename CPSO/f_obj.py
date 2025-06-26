import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
from tqdm import tqdm
from models.lstm_model import LSTMPredictor
import itertools
from utils.train import train_model

def objective_function(position_tensor, train_loader, val_loader, input_size, output_size, device="cpu"):
    losses = []

    for pos in position_tensor:
        num_layers = int(torch.round(pos[0]).clamp(1, 5).item())
        hidden_size = int(torch.round(pos[1]).clamp(16, 256).item())
        lr = float(pos[2].clamp(1e-5, 1e-2).item())
        dropout = float(pos[3].clamp(0.0, 0.6).item())

        model = LSTMPredictor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        val_loss = train_model(
                        model,
                        train_loader,
                        val_loader,
                        n_epochs=10,
                        lr=lr,
                        experiment=None,
                        patience=3
                    )

        losses.append(val_loss)

    return torch.tensor(losses, device=device)

