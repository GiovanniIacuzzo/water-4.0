import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def generate_scenario(
    generator,
    X_cond,
    noise_dim,
    n_scenarios=10,
    batch_size=32,
    save_path=None,
    plot_example=True
):
    """
    Genera N scenari futuri a partire da uno stato iniziale X_cond.
    
    Params:
    - generator: modello generatore allenato
    - X_cond: tensor (n_samples, cond_len, n_features)
    - noise_dim: dimensione del vettore z
    - n_scenarios: numero di scenari da generare per ogni X_cond
    - batch_size: batch size per generazione
    - save_path: facoltativo, se vuoi salvare gli scenari su file
    - plot_example: se True, plottiamo uno scenario di esempio

    Returns:
    - scenarios: ndarray (n_samples, n_scenarios, forecast_horizon, n_features)
    """

    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    generator = generator.to(device)
    generator.eval()

    X_cond = torch.tensor(X_cond, dtype=torch.float32).to(device)
    n_samples = X_cond.shape[0]
    scenarios = []

    with torch.no_grad():
        for i in tqdm(range(n_scenarios), desc="Generating Scenarios"):
            all_preds = []

            for i_start in range(0, n_samples, batch_size):
                x_batch = X_cond[i_start:i_start + batch_size]
                current_bs = x_batch.size(0)
                z = torch.randn(current_bs, noise_dim).to(device)
                y_pred = generator(x_batch, z)
                all_preds.append(y_pred.cpu())

            scenarios.append(torch.cat(all_preds, dim=0).unsqueeze(1))  # (n_samples, 1, forecast_horizon, n_features)

    scenarios = torch.cat(scenarios, dim=1).numpy()  # (n_samples, n_scenarios, T, F)

    if save_path:
        np.save(save_path, scenarios)

    if plot_example:
        n_plot = min(5, n_scenarios)
        idx = np.random.randint(0, n_samples)
        plt.figure(figsize=(12, 6))
        for i in range(n_plot):
            plt.plot(scenarios[idx, i, :, 0], label=f"Scenario {i+1}")
        plt.title(f"Generated Scenarios (sample idx = {idx})")
        plt.xlabel("Time Step")
        plt.ylabel("Forecast Variable")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("generated_scenarios_example.png")
        plt.close()

    return scenarios
