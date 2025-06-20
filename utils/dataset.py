import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class LTownLeakageDataset(Dataset):
    def __init__(
        self,
        demand_files,
        flow_files,
        level_files,
        pressure_files,
        leakage_files,
        seq_len=24,
        task_type="regression"
    ):
        def load_and_concat(files):
            dfs = []
            for path in files:
                df = pd.read_csv(path, sep=";", decimal=",", parse_dates=[0])
                df.set_index(df.columns[0], inplace=True)
                dfs.append(df)
            return pd.concat(dfs)

        demand_df = load_and_concat(demand_files)
        flow_df = load_and_concat(flow_files)
        level_df = load_and_concat(level_files)
        pressure_df = load_and_concat(pressure_files)
        leakage_df = load_and_concat(leakage_files)

        all_inputs = pd.concat([demand_df, flow_df, level_df, pressure_df], axis=1)

        if task_type == "classification":
            target = (leakage_df.sum(axis=1) > 0).astype(int).to_frame(name="leakage")
        elif task_type == "regression":
            target = leakage_df.sum(axis=1).to_frame(name="leakage")
        else:
            raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

        full_df = all_inputs.join(target, how="inner").dropna()

        self.seq_len = seq_len
        self.task_type = task_type

        self.input_scaler = StandardScaler()
        self.inputs = self.input_scaler.fit_transform(full_df.drop(columns=["leakage"]))

        if task_type == "regression":
            self.target_scaler = StandardScaler()
            self.targets = self.target_scaler.fit_transform(full_df[["leakage"]])
        else:
            self.targets = full_df["leakage"].values.reshape(-1, 1)

    def __len__(self):
        return len(self.inputs) - self.seq_len

    def __getitem__(self, idx):
        x = self.inputs[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
