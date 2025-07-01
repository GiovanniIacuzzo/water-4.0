import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
from tqdm import tqdm
from models.lstm_model import LSTMPredictor
import torch.nn as nn
import numpy as np

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

        val_loss = train_model_optimization(
                        model,
                        train_loader,
                        val_loader,
                        n_epochs=1,
                        lr=lr,
                        experiment=None,
                        patience=1
                    )

        losses.append(val_loss)

    return torch.tensor(losses, dtype=torch.float32, device=device)

def train_model_optimization(model, train_loader, val_loader, n_epochs=10, lr=1e-3, patience=3, experiment=None):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{n_epochs}", leave=False)
        for x_batch, y_batch in progress_bar:
            x_batch = x_batch.to(device)  # [B, T, in]
            y_batch = y_batch.to(device)  # [B, T, out]

            optimizer.zero_grad()
            y_pred = model(x_batch)  # [B, T, out]
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                val_loss = loss_fn(y_pred, y_val)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        if experiment:
            experiment.log_metric("val_loss", avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_loss
