import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions(model, loader, scaler, device='cpu', n_samples=100):
    model.eval()
    X, Y, Yhat = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            Yhat.append(y_pred)
            Y.append(y.numpy())
            if len(Yhat) * x.shape[0] > n_samples:
                break

    Yhat = scaler.inverse_transform(np.vstack(Yhat))
    Y = scaler.inverse_transform(np.vstack(Y))

    plt.figure(figsize=(10, 4))
    plt.plot(Y, label="Reale")
    plt.plot(Yhat, label="Predetto")
    plt.title("Confronto tra valori reali e predetti")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()