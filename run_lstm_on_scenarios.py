import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from models.lstm_model import LSTMPredictor

# Restrict thread usage to avoid CPU contention
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"


def load_model(path: str, config: dict, device: str = "cpu") -> torch.nn.Module:
    """
    Load an LSTM model from a .pth checkpoint.
    """
    model = LSTMPredictor(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    checkpoint = torch.load(path, map_location=torch.device(device))
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    if missing:
        print(f"[⚠️] Missing layers: {missing}")
    if unexpected:
        print(f"[⚠️] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def predict_scenarios(
    model: torch.nn.Module,
    scenarios: np.ndarray,
    input_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict leakage given input scenarios using the provided LSTM model.
    """
    assert scenarios.ndim == 4, f"[❌] Expected shape (N, A, T, F), got {scenarios.shape}"
    
    averaged = scenarios.mean(axis=1)  # (N, T, F)
    assert averaged.shape[2] == input_size, \
        f"[❌] Feature mismatch: expected {input_size}, got {averaged.shape[2]}"

    inputs = torch.tensor(averaged, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs).squeeze(-1).cpu().numpy()

    return outputs, averaged


def plot_predictions_vs_observed(
    averaged: np.ndarray,
    predictions: np.ndarray,
    save_path: str = "prediction_vs_observed.png",
    feature_idx: int = 0,  # indice della feature da considerare come "osservata"
    num_scenarios: int = 3
):
    """
    Confronta predizioni e osservazioni (feature reale o media feature) per alcuni scenari.
    """
    assert predictions.ndim == 2, f"Expected predictions shape (N, T), got {predictions.shape}"
    assert averaged.shape[:2] == predictions.shape, "Mismatch tra averaged e predictions"

    for i in range(min(num_scenarios, averaged.shape[0])):
        observed = averaged[i, :, feature_idx]  # ad es. leakage o pressione
        predicted = predictions[i, :]

        plt.figure(figsize=(10, 4))
        plt.plot(observed, label=f"Osservato (feature {feature_idx})", color="black")
        plt.plot(predicted, label="Predetto", color="blue", linestyle="--")
        plt.title(f"Scenario {i} - Predizione vs Osservazione")
        plt.xlabel("Timestep")
        plt.ylabel("Valore Normalizzato")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        scenario_plot_path = f"{save_path.replace('.png', '')}_scenario_{i}.png"
        plt.savefig(scenario_plot_path)
        plt.show()
        print(f"[🖼️] Salvato: {scenario_plot_path}")

        # Errori
        mae = np.mean(np.abs(predicted - observed))
        rmse = np.sqrt(np.mean((predicted - observed) ** 2))
        print(f"[📊] Scenario {i} → MAE = {mae:.4f}, RMSE = {rmse:.4f}")



def main():
    # === Configuration ===
    model_config = {
        "input_size": 119,
        "output_size": 1,
        "hidden_size": 195,    # ← aumentato
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True  # ← se usato
    }

    paths = {
        "model": "best_lstm_model.pth",
        "scenarios": "scenarios.npy",
        "output_predictions": "predicted_leakages.npy",
        "plot": "leakage_scenarios.png"
    }

    # === Device Detection ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Using device: {device}")

    # === Model Loading ===
    print("[📥] Loading model...")
    model = load_model(paths["model"], model_config, device)

    # === Scenario Loading ===
    print("[📥] Loading synthetic scenarios...")
    if not Path(paths["scenarios"]).exists():
        raise FileNotFoundError(f"[❌] Scenario file not found: {paths['scenarios']}")

    scenarios = np.load(paths["scenarios"])
    print(f"[✅] Loaded scenarios: {scenarios.shape}")

    # === Inference ===
    print("[🔍] Running inference on scenarios...")
    predictions, averaged = predict_scenarios(model, scenarios, model_config["input_size"])
    print(f"[📈] Sample predictions: {predictions[:5]}")

    # === Statistical Analysis ===
    var_feat = np.var(averaged, axis=(0, 1)).mean()
    var_scen = np.var(averaged, axis=1).mean()
    print(f"[📊] Feature-wise mean variance: {var_feat:.6f}")
    print(f"[📊] Scenario-wise mean variance: {var_scen:.6f}")
    print(f"[📈] Pred. stats: min={predictions.min():.4f}, max={predictions.max():.4f}, std={predictions.std():.4f}")

    if np.allclose(predictions, predictions[0]):
        print("[⚠️] All predictions are equal:", predictions[0])
    elif np.isnan(predictions).any():
        print("[❌] Predictions contain NaN!")
    else:
        print("[✅] Prediction variation detected.")

    # === Save Outputs ===
    np.save(paths["output_predictions"], predictions)
    print(f"[💾] Predictions saved to: {paths['output_predictions']}")

    # === Visualization ===
    plot_predictions_vs_observed(
    averaged,
    predictions,
    save_path=paths["plot"],
    feature_idx=0,  # cambia se vuoi un'altra feature
    num_scenarios=5  # quanti scenari plottare
)


if __name__ == "__main__":
    main()
