import torch
import numpy as np
import pandas as pd
import os
from models import build_models
from dataset import LeakScenarioDataset
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_DIM = 16
MODEL_DIR = "checkpoints"
N_SAMPLES = 1000  # Numero di campioni da generare

def generate_and_evaluate():
    print("🔍 Avvio generazione scenari di perdita...")

    # === Dataset (per caricare encoders e scaler)
    dataset = LeakScenarioDataset(csv_path='data/leak_data.csv')
    scaler, _, type_enc, weekday_enc, month_enc = dataset.get_scalers()

    # === Modelli
    G, _, cond_embedder = build_models(NOISE_DIM)
    G.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_generator.pth"), map_location=DEVICE))
    cond_embedder.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_condition_embedder.pth"), map_location=DEVICE))
    G, cond_embedder = G.to(DEVICE), cond_embedder.to(DEVICE)
    G.eval(), cond_embedder.eval()

    # === Definizione condizioni da testare (puoi espandere con un CSV, input o loop)
    fixed_conditions = {
        'weekday': 'Monday',
        'month': 'July',
        'leak_type': 'burst',
        'start_time': 10  # ora: 10:00
    }

    # === Encoding condizioni
    weekday = torch.tensor([weekday_enc.transform([fixed_conditions['weekday']])[0]] * N_SAMPLES, dtype=torch.long).to(DEVICE)
    month = torch.tensor([month_enc.transform([fixed_conditions['month']])[0]] * N_SAMPLES, dtype=torch.long).to(DEVICE)
    leak_type = torch.tensor([type_enc.transform([fixed_conditions['leak_type']])[0]] * N_SAMPLES, dtype=torch.long).to(DEVICE)
    start_time = torch.tensor([fixed_conditions['start_time']] * N_SAMPLES, dtype=torch.long).to(DEVICE)

    cond_vec = cond_embedder(weekday, month, leak_type, start_time)

    # === Generazione
    z = torch.randn(N_SAMPLES, NOISE_DIM).to(DEVICE)
    with torch.no_grad():
        fake_data = G(z, cond_vec).cpu().numpy()

    # === Inverse transform
    fake_data_original = scaler.inverse_transform(fake_data)
    durations = fake_data_original[:, 0]
    severities = fake_data_original[:, 1]

    # === Output DataFrame
    results = pd.DataFrame({
        'duration': durations,
        'severity': severities,
        'weekday': fixed_conditions['weekday'],
        'month': fixed_conditions['month'],
        'leak_type': fixed_conditions['leak_type'],
        'start_time': fixed_conditions['start_time']
    })

    # === Salvataggio
    out_path = "generated_samples.csv"
    results.to_csv(out_path, index=False)
    print(f"✅ {N_SAMPLES} campioni generati e salvati in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate"], default="generate")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_and_evaluate()
