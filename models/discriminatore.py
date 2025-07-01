import torch
import torch.nn as nn

class DiscriminatorCGAN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len_total, dropout=0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x_full):
        # x_full: [B, seq_len_total, input_size] = [X_cond || X_target]
        out, _ = self.lstm(x_full)  # [B, seq_len_total, H*2]
        h_last = out[:, -1, :]      # Prendi ultimo hidden state (oppure media)
        score = self.classifier(h_last)  # [B, 1]
        return score
