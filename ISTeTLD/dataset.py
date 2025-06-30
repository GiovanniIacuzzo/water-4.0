import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime


class LeakScenarioDataset(Dataset):
    def __init__(self, csv_path, transform=True):
        """
        Dataset per la CGAN che genera scenari di perdita realistici.

        Args:
            csv_path (str): Path al file CSV contenente i dati.
            transform (bool): Se True, normalizza/standardizza i dati.
        """
        self.transform = transform
        self.data = pd.read_csv(csv_path)

        # === Verifica colonne ===
        required_cols = ['node_id', 'start_time', 'duration', 'severity', 'leak_type', 'weekday', 'month']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Colonne mancanti nel CSV: {missing}")

        # === Preprocessing ===
        self._preprocess()

    def _preprocess(self):
        df = self.data.copy()

        # Parsing start_time ('HH:MM') → ora (int)
        df['start_time'] = df['start_time'].apply(self._parse_hour)

        # Encoding categoriali
        self.node_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.weekday_encoder = LabelEncoder()
        self.month_encoder = LabelEncoder()

        df['node_id'] = self.node_encoder.fit_transform(df['node_id'])
        df['leak_type'] = self.type_encoder.fit_transform(df['leak_type'])
        df['weekday'] = self.weekday_encoder.fit_transform(df['weekday'])
        df['month'] = self.month_encoder.fit_transform(df['month'])

        # === Feature e condizioni per la CGAN ===
        self.feature_cols = ['duration', 'severity']
        self.condition_cols = ['weekday', 'month', 'leak_type', 'start_time']

        self.features = df[self.feature_cols].astype(np.float32)
        self.conditions = df[self.condition_cols].astype(np.float32)

        if self.transform:
            self.scaler = StandardScaler()
            self.features[self.feature_cols] = self.scaler.fit_transform(self.features[self.feature_cols])

        # === Tensori finali ===
        self.features_tensor = torch.tensor(self.features.values, dtype=torch.float32)
        self.condition_tensor = torch.tensor(self.conditions.values, dtype=torch.float32)

    def _parse_hour(self, time_str):
        """Converte 'HH:MM' → int (es. '14:00' → 14)."""
        try:
            return int(datetime.datetime.strptime(time_str.strip(), "%H:%M").hour)
        except Exception as e:
            raise ValueError(f"Errore parsing orario '{time_str}': {e}")

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        return {
            'duration': self.features_tensor[idx, 0:1],
            'severity': self.features_tensor[idx, 1:2],
            'weekday': self.condition_tensor[idx, 0:1].long().squeeze(0),
            'month': self.condition_tensor[idx, 1:2].long().squeeze(0),
            'leak_type': self.condition_tensor[idx, 2:3].long().squeeze(0),
            'start_time': self.condition_tensor[idx, 3:4].long().squeeze(0)
        }

    def get_scalers(self):
        """Restituisce gli scaler per usare inverse_transform in test/generazione."""
        return self.scaler, self.node_encoder, self.type_encoder, self.weekday_encoder, self.month_encoder
