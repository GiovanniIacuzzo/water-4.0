import torch
from tqdm import tqdm
from models.lstm_model import LSTMPredictor
import itertools

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

        sampled_loader = list(itertools.islice(train_loader, 10))
        val_loss = train_model_f(model, sampled_loader, n_epochs=5, lr=lr)

        losses.append(val_loss)

    return torch.tensor(losses, device=device)


def train_model_f(model, train_loader, n_epochs=1, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss
