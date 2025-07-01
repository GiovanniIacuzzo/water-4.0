import torch
import torch.nn as nn

class GeneratorCGAN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, noise_dim, dropout=0.4):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.noise_dim = noise_dim

        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + noise_dim, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, forecast_horizon * input_size)  # output futuro multivariato
        )

    def forward(self, x_cond, z):
        # x_cond: [batch_size, seq_len, input_size]
        # z: [batch_size, noise_dim]
        batch_size = x_cond.size(0)

        _, (h_n, _) = self.lstm_encoder(x_cond)  # [num_layers*2, B, H]
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, H*2]

        context_and_noise = torch.cat([h_last, z], dim=1)  # [B, H*2 + noise_dim]

        out = self.decoder(context_and_noise)  # [B, forecast_horizon * input_size]
        out = out.view(batch_size, self.forecast_horizon, -1)  # [B, forecast_horizon, input_size]
        return out
