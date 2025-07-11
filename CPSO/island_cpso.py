import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import threading
from rich.console import Console

console = Console()

def logger_worker(queue):
    while True:
        msg = queue.get()
        if msg == "STOP":
            break
        console.log(msg)

def island_cpso(train_loader, val_loader, input_size, output_size,
                dim=4, lb=None, ub=None,
                num_islands=4, migrations=3, migration_interval=5,
                options=None, device="cpu"):

    lb = np.array(lb or [1, 16, 1e-5, 0.0])
    ub = np.array(ub or [5, 256, 1e-2, 0.6])

    manager = mp.Manager()
    return_dict = manager.dict()
    best_global = manager.dict()
    log_queue = manager.Queue()
    migration_pool = manager.dict()
    barrier = mp.Barrier(num_islands)

    log_thread = threading.Thread(target=logger_worker, args=(log_queue,), daemon=True)
    log_thread.start()

    console.rule("[bold cyan]AVVIO CPSO - MODELLO A ISOLE")

    total_particles = options.get('particles', 4)
    particles_per_island = max(1, total_particles // num_islands)

    for mig in range(migrations):
        log_queue.put(f"[yellow]\n[Migrazione {mig + 1}/{migrations}] Round di ottimizzazione in corso...")

        processes = []
        for i in range(num_islands):
            log_queue.put(f"[Setup] Inizializzo Isola {i}")

            local_lb = lb + i * (ub - lb) / num_islands
            local_ub = lb + (i + 1) * (ub - lb) / num_islands

            local_options = options.copy() if options else {}
            local_options['particles'] = particles_per_island

            sub_interval = options.get('sub_interval', 5)
            p = mp.Process(target=optimize_in_island,
                           args=(i, return_dict, best_global, train_loader, val_loader,
                                 input_size, output_size, local_options,
                                 local_lb.tolist(), local_ub.tolist(),
                                 dim, device, sub_interval, log_queue, barrier, migration_pool, num_islands))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        valid_candidates = [v for v in return_dict.values() if v['best_cost'] != float('inf')]
        if not valid_candidates:
            raise RuntimeError("Nessuna isola ha restituito un risultato valido.")

        best_candidate = min(valid_candidates, key=lambda x: x['best_cost'])
        best_global['pos'] = best_candidate['best_pos']
        best_global['cost'] = best_candidate['best_cost']

        log_queue.put(f"[green][Migrazione {mig + 1}] Miglior costo globale: {best_global['cost']:.6f}")

    log_queue.put("STOP")
    log_thread.join()

    # === GRAFICO MIGLIORATO ===
    plt.figure(figsize=(10, 6))
    for island_id in sorted(return_dict.keys()):
        data = return_dict[island_id]
        history = data.get("history", [])

        if isinstance(history, torch.Tensor):
            history = history.cpu().numpy()
        elif not isinstance(history, (list, np.ndarray)):
            history = []

        if len(history) > 0:
            plt.plot(history, label=f"Isola {island_id}", marker='o')
        else:
            log_queue.put(f"[yellow][Avviso] Isola {island_id} ha history vuota: curva non tracciata.")

    plt.title("Curve di Convergenza CPSO - Modello a Isole")
    plt.xlabel("Iterazioni")
    plt.ylabel("Costo minimo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergenza_isole.png")
    console.print("[bold green] Curva di convergenza salvata in 'convergenza_isole.png'")

    best_params = best_global['pos']
    num_layers = int(round(best_params[0]))
    hidden_size = int(round(best_params[1]))
    lr = float(best_params[2])
    dropout = float(best_params[3])

    return num_layers, hidden_size, lr, dropout

def optimize_in_island(island_id, return_dict, best_global, train_loader, val_loader,
                        input_size, output_size, options, lb, ub, dim, device, sub_interval, log_queue, barrier, migration_pool, num_islands):
    from CPSO.CPSO import CPSO
    from CPSO.f_obj import objective_function

    try:
        log_queue.put(f"[blue][Isola {island_id}] Ottimizzazione per {sub_interval} iterazioni")

        def wrapped_obj(x):
            for i in range(len(x)):
                log_queue.put(f"[Isola {island_id}] Valuto particella {i+1}/{len(x)}")
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
            device=device,
            log_queue=log_queue,
            island_id=island_id
        )

        if 'pos' in best_global:
            log_queue.put(f"[cyan][Isola {island_id}] Sincronizzo con best globale iniziale")
            optimizer.global_best_position = torch.tensor(best_global['pos'], device=device)
            optimizer.global_best_cost = float(best_global['cost'])

        best_pos, best_cost, exec_time, history = optimizer.optimize()

        migration_pool[island_id] = {
            'best_pos': best_pos.tolist(),
            'best_cost': best_cost
        }

        log_queue.put(f"[Isola {island_id}] In attesa delle altre isole per la migrazione...")
        barrier.wait()

        immigrants = [v for k, v in migration_pool.items() if k != island_id]
        if immigrants:
            best_immigrant = min(immigrants, key=lambda x: x['best_cost'])
            immigrant_tensor = torch.tensor(best_immigrant['best_pos'], device=device)
            log_queue.put(f"[Isola {island_id}] Migrazione: importato best da altra isola con costo {best_immigrant['best_cost']:.4f}")
            optimizer.global_best_position = immigrant_tensor
            optimizer.global_best_cost = best_immigrant['best_cost']

        log_queue.put(f"[Isola {island_id}] Migrazione sincronizzata completata.")

        return_dict[island_id] = {
            'best_pos': best_pos,
            'best_cost': best_cost,
            'history': history
        }

        log_queue.put(f"[magenta][Isola {island_id}] Fine ottimizzazione - Best Cost: {best_cost:.4f}")

    except Exception as e:
        log_queue.put(f"[red][Errore][Isola {island_id}] {str(e)}")
        return_dict[island_id] = {
            'best_pos': None,
            'best_cost': float('inf'),
            'history': []
        }