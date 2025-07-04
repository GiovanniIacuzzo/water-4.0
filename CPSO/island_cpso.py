import multiprocessing as mp

def island_cpso(train_loader, val_loader, input_size, output_size,
                dim=4, lb=None, ub=None,
                num_islands=4, migrations=3, migration_interval=5,
                options=None, device="cpu"):

    lb = lb or [1, 16, 1e-5, 0.0]
    ub = ub or [5, 256, 1e-2, 0.6]

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(num_islands):
        p = mp.Process(target=optimize_in_island,
                       args=(i, return_dict, train_loader, val_loader,
                             input_size, output_size, options, lb, ub, dim, device))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    best_overall = min(return_dict.values(), key=lambda x: x['best_cost'])
    best_params = best_overall['best_pos']

    num_layers = int(round(best_params[0]))
    hidden_size = int(round(best_params[1]))
    lr = float(best_params[2])
    dropout = float(best_params[3])

    return num_layers, hidden_size, lr, dropout

def optimize_in_island(island_id, return_dict, train_loader, val_loader, input_size, output_size, options, lb, ub, dim, device):
    from CPSO.CPSO import CPSO
    from CPSO.f_obj import objective_function

    def wrapped_obj(x):
        return objective_function(x, train_loader, val_loader, input_size, output_size, device=device)

    local_options = options.copy() if options else {}
    local_options['log_file'] = f'cpso_island_{island_id}.csv'

    optimizer = CPSO(
        objective_fn=wrapped_obj,
        dim=dim,
        lb=lb,
        ub=ub,
        options=local_options,
        device=device
    )

    best_pos, best_cost, exec_time, history = optimizer.optimize()
    return_dict[island_id] = {
        'best_pos': best_pos,
        'best_cost': best_cost,
        'history': history
    }