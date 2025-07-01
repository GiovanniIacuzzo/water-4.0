import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.lstm_model import LSTMPredictor

def load_model(path, input_size, output_size, hidden_size, num_layers, dropout):
    model = LSTMPredictor(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("Layer mancanti nel checkpoint:", missing)
    if unexpected:
        print("Layer inattesi nel checkpoint:", unexpected)

    model.eval()
    return model

if __name__ == "__main__":
    model_config = {
        "input_size": 119,
        "output_size": 1,
        "hidden_size": 42,
        "num_layers": 2,
        "dropout": 0.3
    }

    # === Caricamento modello ===
    model = load_model("best_lstm_model.pth", **model_config)

    # === Caricamento scenari ===
    scenarios = np.load("scenarios.npy")  # shape: (N, A, T, F)
    print("Scenari caricati:", scenarios.shape)

    if scenarios.ndim != 4:
        raise ValueError(f"Attesa shape (N, A, T, F), ma trovato: {scenarios.shape}")

    averaged = scenarios.mean(axis=1)  # shape: (N, T, F)
    print("Shape dopo media sugli assi:", averaged.shape)

    if averaged.shape[2] != model_config["input_size"]:
        raise ValueError(f"Feature mismatch: input_size={model_config['input_size']}, trovato={averaged.shape[2]}")

    # === Conversione in tensore ===
    inputs = torch.tensor(averaged, dtype=torch.float32)

    # === Previsione ===
    with torch.no_grad():
        outputs = model(inputs)  # shape: (N, 1)
    print("Output raw shape:", outputs.shape)

    predictions = outputs.squeeze(-1).numpy()  # shape: (N,)
    print("Prime 5 predizioni:", predictions[:5])

    if np.allclose(predictions, predictions[0]):
        print("Tutte le predizioni sono uguali:", predictions[0])
    elif np.isnan(predictions).any():
        print("Le predizioni contengono NaN!")
    else:
        print("Predizioni variegate")

    for i in range(3):
        plt.plot(averaged[i].flatten(), label=f"Scenario {i}")
    plt.title("Input (flattened) per primi 3 scenari")
    plt.legend()
    plt.show()

    print("📊 Varianza media per feature:", np.var(averaged, axis=(0, 1)).mean())
    print("📊 Varianza media per scenario:", np.var(averaged, axis=1).mean())
    print("📈 Predizioni: min =", predictions.min(), "max =", predictions.max(), "std =", predictions.std())

    # === Salvataggio ===
    np.save("predicted_leakages.npy", predictions)
    print("Predizioni salvate in predicted_leakages.npy")

    # === Visualizzazione ===
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(predictions))

    plt.bar(indices, predictions, color='skyblue')
    plt.title("Leakage previsto per ciascuno scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Leakage")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("leakage_scenarios.png")
    plt.show()
