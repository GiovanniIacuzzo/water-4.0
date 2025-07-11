import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
from models.lstm_model import LSTMPredictor
import torch.nn as nn
import numpy as np

def objective_function(position_tensor, train_loader, val_loader, input_size, output_size, device="cpu"):
    losses = []
    max_batches = 3  # batch per epoca

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

        # Versione semplificata: solo pochi batch
        val_loss = train_evaluation(
            model,
            train_loader,
            val_loader,
            lr=lr,
            device=device,
            max_batches=max_batches
        )
        losses.append(val_loss)

    return torch.tensor(losses, dtype=torch.float32, device=device)


def train_evaluation(model, train_loader, val_loader, lr=1e-3, device="cpu", max_batches=3):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    batch_count = 0

    for x_batch, y_batch in train_loader:
        if batch_count >= max_batches:
            break
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        batch_count += 1

    # Valutazione leggera
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_loader):
            if i >= max_batches:
                break
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_pred = model(x_val)
            val_loss = loss_fn(y_pred, y_val)
            val_losses.append(val_loss.item())

    return np.mean(val_losses)
