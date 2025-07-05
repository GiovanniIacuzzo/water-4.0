import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
from CPSO.island_cpso import island_cpso

def optimize_with_cpso(train_loader, val_loader, input_size, output_size, experiment=None):
    # Parametri standard
    dim = 4
    lb = [1, 16, 1e-5, 0.0]
    ub = [5, 256, 1e-2, 0.6]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    options = {
        'particles': 2,
        'sub_interval': 2,
        'mu_max': 0.9,
        'mu_min': 0.4,
        'dt': 2,
        'Cognitive_constant': 2.05,
        'Social_constant': 2.05,
        'maxNoChange': 2,
        'tol': 1e-4,
        'print_every': 1
    }

    return island_cpso(
        train_loader, val_loader, input_size, output_size,
        dim=dim, lb=lb, ub=ub,
        num_islands=2,
        migrations=1,
        migration_interval=0,
        options=options,
        device=device
    )