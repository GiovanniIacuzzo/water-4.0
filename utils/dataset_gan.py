import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class GANLeakageDataset(Dataset):
    def __init__(
        self,
        demand_files,
        flow_files,
        level_files,
        pressure_files,
        seq_len=288,
        forecast_horizon=48,
    ):
        def load_and_concat(files):
            dfs = []
            for path in files:
                df = pd.read_csv(path, sep=";", decimal=",", parse_dates=[0])
                df.set_index(df.columns[0], inplace=True)
                dfs.append(df)
            return pd.concat(dfs)

        # Caricamento e fusione dei dati
        demand_df = load_and_concat(demand_files)
        flow_df = load_and_concat(flow_files)
        level_df = load_and_concat(level_files)
        pressure_df = load_and_concat(pressure_files)

        # Dati input multivariati
        all_inputs = pd.concat([demand_df, flow_df, level_df, pressure_df], axis=1)
        full_df = all_inputs.dropna()

        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Normalizzazione
        self.scaler = StandardScaler()
        self.inputs = self.scaler.fit_transform(full_df)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs) - self.seq_len - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x_cond = self.inputs[idx:idx + self.seq_len]  # [seq_len, n_features]
        x_target = self.inputs[idx + self.seq_len:idx + self.seq_len + self.forecast_horizon]  # [forecast_horizon, n_features]
        return x_cond, x_target
