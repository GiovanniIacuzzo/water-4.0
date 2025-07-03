import torch.nn as nn
import torch

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden*2)
        batch_size, seq_len, hidden_dim = out.shape

        # Applica fc a ciascun timestep in modo indipendente
        out = self.fc(out.contiguous().view(-1, hidden_dim))  # (batch * seq_len, output_size)
        out = out.view(batch_size, seq_len, -1)  # (batch, seq_len, output_size)

        return out
