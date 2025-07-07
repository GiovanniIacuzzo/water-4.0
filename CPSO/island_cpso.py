# Modulo aggiornato: CPSO/island_cpso.py

import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

def island_cpso(train_loader, val_loader, input_size, output_size,
                dim=4, lb=None, ub=None,
                num_islands=4, migrations=3, migration_interval=5,
                options=None, device="cpu"):

    lb = np.array(lb or [1, 16, 1e-5, 0.0])
    ub = np.array(ub or [5, 256, 1e-2, 0.6])

    manager = mp.Manager()
    return_dict = manager.dict()
    best_global = manager.dict()

    console.rule("[bold cyan]AVVIO CPSO - MODELLO A ISOLE")

    # Calcolo delle particelle per isola
    total_particles = options.get('particles', 4)
    particles_per_island = max(1, total_particles // num_islands)

    for mig in range(migrations):
        console.print(f"[bold yellow]\n[Migrazione {mig + 1}/{migrations}] Round di ottimizzazione in corso...")

        processes = []
        for i in range(num_islands):
            console.print(f"[Setup] Inizializzo Isola {i}")

            # Suddivisione dello spazio di ricerca
            local_lb = lb + i * (ub - lb) / num_islands
            local_ub = lb + (i + 1) * (ub - lb) / num_islands

            # Opzioni locali con override del numero di particelle
            local_options = options.copy() if options else {}
            local_options['particles'] = particles_per_island

            sub_interval = options.get('sub_interval', 5)
            p = mp.Process(target=optimize_in_island,
                           args=(i, return_dict, best_global, train_loader, val_loader,
                                 input_size, output_size, local_options,
                                 local_lb.tolist(), local_ub.tolist(),
                                 dim, device, sub_interval))

            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Dopo ogni migrazione aggiorna il miglior global
        best_candidate = min(return_dict.values(), key=lambda x: x['best_cost'])
        best_global['pos'] = best_candidate['best_pos']
        best_global['cost'] = best_candidate['best_cost']

        console.print(f"[green][Migrazione {mig + 1}] Miglior costo globale: {best_global['cost']:.6f}")

        # === Plot curve di convergenza per ogni isola ===
    plt.figure(figsize=(10, 6))
    for island_id, data in return_dict.items():
        history = data["history"]
        if isinstance(history, torch.Tensor):
            history = history.cpu().numpy()
        plt.plot(history, label=f"Isola {island_id}")

    plt.title("Curve di Convergenza CPSO - Modello a Isole")
    plt.xlabel("Iterazioni")
    plt.ylabel("Costo minimo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergenza_isole.png")
    console.print("[bold green]✅ Curva di convergenza salvata in 'convergenza_isole.png'")

    # === Estrai migliori iperparametri ===
    best_params = best_global['pos']
    num_layers = int(round(best_params[0]))
    hidden_size = int(round(best_params[1]))
    lr = float(best_params[2])
    dropout = float(best_params[3])

    return num_layers, hidden_size, lr, dropout


def optimize_in_island(island_id, return_dict, best_global, train_loader, val_loader,
                        input_size, output_size, options, lb, ub, dim, device, sub_interval):
    from CPSO.CPSO import CPSO
    from CPSO.f_obj import objective_function

    console = Console()
    console.print(f"[blue][Isola {island_id}] Ottimizzazione per {sub_interval} iterazioni")

    def wrapped_obj(x):
        for i in range(len(x)):
            console.print(f"[Isola {island_id}] Valuto particella {i+1}/{len(x)}")
        return objective_function(x, train_loader, val_loader, input_size, output_size, device=device)

    local_options = options.copy() if options else {}
    local_options['log_file'] = f'cpso_island_{island_id}.csv'
    local_options['sub_interval'] = sub_interval

    optimizer = CPSO(
        objective_fn=wrapped_obj,
        dim=dim,
        lb=lb,
        ub=ub,
        options=local_options,
        device=device
    )

    # Se c'è un global best, usalo
    if 'pos' in best_global:
        console.print(f"[cyan][Isola {island_id}] Sincronizzo con best globale iniziale")
        optimizer.global_best_position = torch.tensor(best_global['pos'], device=device)
        optimizer.global_best_cost = float(best_global['cost'])

    best_pos, best_cost, exec_time, history = optimizer.optimize()
    console.print(f"[magenta][Isola {island_id}] Fine ottimizzazione - Best Cost: {best_cost:.4f}")

    return_dict[island_id] = {
        'best_pos': best_pos,
        'best_cost': best_cost,
        'history': history
    }