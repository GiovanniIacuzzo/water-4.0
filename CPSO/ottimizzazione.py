import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
import matplotlib.pyplot as plt
import numpy as np
from CPSO.f_obj import objective_function
from CPSO.CPSO import CPSO

def optimize_with_cpso(train_loader, val_loader, input_size, output_size, experiment=None):
    dim = 4
    lb = [1, 16, 1e-5, 0.0]
    ub = [5, 256, 1e-2, 0.6]

    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilizzato: {device}")

    def wrapped_objective_tensor(x_tensor):
        return objective_function(
            x_tensor,
            train_loader,
            val_loader,
            input_size,
            output_size,
            device=device
        )

    # Opzioni custom per CPSO, personalizzabili
    options = {
        'particles': 20,
        'sub_interval': 20,
        'mu_max': 0.9,
        'mu_min': 0.4,
        'dt': 3,
        'Cognitive_constant': 2.05,
        'Social_constant': 2.05,
        'maxNoChange': 5,
        'tol': 1e-6,
        'max_velocity': None,
        'print_every': 1,
        'log_file': 'cpso_log.csv'
    }

    # Inizializza CPSO
    optimizer = CPSO(
        objective_fn=wrapped_objective_tensor,
        dim=dim,
        lb=lb,
        ub=ub,
        options=options,
        device=device
    )

    best_params, best_cost, exec_time, cost_history = optimizer.optimize()
    print(f"Best Parameters: {best_params}")

    # Interpreta i parametri ottimizzati
    num_layers = int(round(best_params[0]))
    hidden_size = int(round(best_params[1]))
    lr = float(best_params[2])
    dropout = float(best_params[3])

    # Tracciamento curva di convergenza
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history, label='Best Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('CPSO Convergence Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if experiment:
        experiment.log_figure(figure_name="CPSO Convergence Curve", figure=plt)

    plt.savefig("cpso_convergence.png")

    return num_layers, hidden_size, lr, dropout
