import torch
import time
import csv
import os
import matplotlib.pyplot as plt
from typing import Callable, Union
from torch import exp, sqrt
from tqdm import trange

class CPSO:
    def __init__(self,
                 objective_fn: Callable,
                 dim: int,
                 lb: Union[float, list],
                 ub: Union[float, list],
                 options: dict = None,
                 device: str = 'mps'):

        default_options = {
            'particles': 100,
            'sub_interval': 100,
            'mu_max': 0.9,
            'mu_min': 0.4,
            'dt': 0.1,
            'Cognitive_constant': 2.05,
            'Social_constant': 2.05,
            'maxNoChange': 10,
            'tol': 1e-6,
            'max_velocity': None,
            'print_every': 1,
            'log_file': 'cpso_log.csv'
        }
        if options is None:
            options = default_options
        else:
            default_options.update(options)
            options = default_options

        self.objective_fn = objective_fn
        self.dim = dim
        self.device = device
        self.particles = options['particles']
        self.sub_interval = options['sub_interval']
        self.mu_max = options['mu_max']
        self.mu_min = options['mu_min']
        self.dt = options['dt']
        self.Cc = options['Cognitive_constant']
        self.Sc = options['Social_constant']
        self.maxNoChange = options['maxNoChange']
        self.tol = options['tol']
        self.max_velocity = options['max_velocity']
        self.print_every = options['print_every']
        self.log_file = options['log_file']

        self.interval = self.sub_interval * self.dt

        self.VarMin = torch.full((dim,), lb, device=device) if isinstance(lb, (float, int)) else torch.tensor(lb, device=device)
        self.VarMax = torch.full((dim,), ub, device=device) if isinstance(ub, (float, int)) else torch.tensor(ub, device=device)

        self.positions = self.VarMin + (self.VarMax - self.VarMin) * torch.rand((self.particles, dim), device=device)
        self.velocities = 0.1 * (2 * torch.rand((self.particles, dim), device=device) - 1)
        self.costs = self.objective_fn(self.positions)

        self.best_positions = self.positions.clone()
        self.best_costs = self.costs

        global_best_idx = torch.argmin(self.costs)
        self.global_best_position = self.positions[global_best_idx].clone()
        self.global_best_cost = self.costs[global_best_idx].item()

        self.BestCost = []
        self.epsilon = 1e-2

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'best_cost'])

    def optimize(self):
        self.noChangeCount = 0
        self.restored_iteration = 0
        start_time = time.time()
        if os.path.exists("cpso_checkpoint.pt"):
            print("Checkpoint trovato. Provo a ripristinare lo stato...")
            try:
                self.load_checkpoint("cpso_checkpoint.pt")
            except Exception as e:
                print(f"Errore durante il ripristino del checkpoint: {e}")
                print("Il file è probabilmente corrotto. Lo elimino e riparto da zero.")
                os.remove("cpso_checkpoint.pt")

        for it in trange(self.restored_iteration, self.sub_interval, desc="CPSO Optimization"):
            r1 = torch.rand((self.particles, 1), device=self.device)
            r2 = torch.rand((self.particles, 1), device=self.device)
            omega2 = self.Cc * r1 + self.Sc * r2
            omega = sqrt(omega2)
            mu = self.mu_max - ((self.mu_max - self.mu_min) / (self.interval)) * (it - 1) * self.dt
            zeta = (1 - mu) / (2 * omega)
            t = (it + 1) * self.dt

            sqrt_term = sqrt(torch.clamp(zeta ** 2 - 1, min=1e-4))
            lambda1 = -omega * (zeta + sqrt_term)
            lambda2 = -omega * (zeta - sqrt_term)

            fk = (self.Cc * r1 * self.best_positions) + (self.Sc * r2 * self.global_best_position.unsqueeze(0))
            fk_omega = (fk / omega2)

            if torch.all(lambda1 == lambda2):
                lambda_val = lambda1
                c1 = (((self.positions - fk_omega) * (1 + lambda_val * t)) - self.velocities * t) 
                c2 = (self.velocities - ((self.positions - fk_omega) * lambda_val))

                new_positions = (c1 + c2 * t) * exp(lambda_val * t) + fk_omega
                new_positions += self.epsilon * torch.randn_like(new_positions)
                new_velocities = c2 * exp(lambda_val * t) + (c1 + c2 * t) * exp(lambda_val * t) * lambda_val
            else:
                c1 = (((self.positions - fk_omega) * lambda2 ) - self.velocities) / ((lambda2 - lambda1))
                c2 = (self.velocities - ((self.positions - fk_omega) * lambda1)) / ((lambda2 - lambda1))

                new_positions = c1 * exp(lambda1 * (t)) + c2 * exp(lambda2 * (t)) + fk_omega
                new_positions += self.epsilon * torch.randn_like(new_positions)
                new_velocities = c1 * lambda1 * exp(lambda1 * (t)) + c2 * lambda2 * exp(lambda2 * (t))

            new_positions = torch.clamp(new_positions, self.VarMin, self.VarMax)
            new_velocities = torch.clamp(new_velocities, 0.1 * self.VarMin, 0.1 * self.VarMax)

            self.positions = new_positions
            # print("Posizioni:", self.positions)
            
            self.velocities = new_velocities
            # print("Velocità:", self.velocities)

            self.costs = self.objective_fn(self.positions)
            # print("Costi:", self.costs)

            improved = self.costs < self.best_costs
            self.best_positions[improved] = self.positions[improved]
            self.best_costs[improved] = self.costs[improved]

            min_cost, min_idx = torch.min(self.costs.view(-1), 0)
            if min_cost.item() < self.global_best_cost:
                self.global_best_cost = min_cost.item()
                self.global_best_position = self.positions[min_idx].clone()
                self.noChangeCount = 0
            else:
                self.noChangeCount += 1

            self.BestCost.append(self.global_best_cost)

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([it + 1, self.global_best_cost])

            if (it + 1) % self.print_every == 0 or it == 0:
                print(f"[Iter {it + 1:03d}] Best Cost: {self.global_best_cost:.6f}")

            self.save_checkpoint("cpso_checkpoint.pt", iteration=it + 1)

            if self.noChangeCount >= self.maxNoChange:
                print("Early stopping...")
                break

        exec_time = time.time() - start_time

        if os.path.exists("cpso_checkpoint.pt"):
            os.remove("cpso_checkpoint.pt")

        return (
            self.global_best_position.cpu().numpy(),
            self.global_best_cost,
            exec_time,
            torch.tensor(self.BestCost).cpu().numpy()
        )


    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.BestCost, label='Best Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('CPSO Optimization Progress')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def batch_optimize(self, runs=10, verbose=True):
        all_best_positions = []
        all_best_costs = []
        all_times = []
        all_histories = []

        for run in range(runs):
            if verbose:
                print(f"\n=== Run {run + 1}/{runs} ===")

            # Re-inizializza lo swarm per ogni run
            self.__init__(
                objective_fn=self.objective_fn,
                dim=self.dim,
                lb=self.VarMin.cpu().numpy().tolist(),
                ub=self.VarMax.cpu().numpy().tolist(),
                options={
                    'particles': self.particles,
                    'sub_interval': self.sub_interval,
                    'mu_max': self.mu_max,
                    'mu_min': self.mu_min,
                    'dt': self.dt,
                    'Cognitive_constant': self.Cc,
                    'Social_constant': self.Sc,
                    'maxNoChange': self.maxNoChange,
                    'tol': self.tol,
                    'max_velocity': self.max_velocity,
                    'print_every': self.print_every,
                    'log_file': self.log_file
                },
                device=self.device
            )

            best_pos, best_cost, exec_time, history = self.optimize()

            all_best_positions.append(best_pos)
            all_best_costs.append(best_cost)
            all_times.append(exec_time)
            all_histories.append(history)

        return {
            'best_positions': all_best_positions,
            'best_costs': all_best_costs,
            'execution_times': all_times,
            'histories': all_histories
        }

    def save_checkpoint(self, path="cpso_checkpoint.pt", iteration=None):
        checkpoint = {
            'positions': self.positions,
            'velocities': self.velocities,
            'costs': self.costs,
            'best_positions': self.best_positions,
            'best_costs': self.best_costs,
            'global_best_position': self.global_best_position,
            'global_best_cost': self.global_best_cost,
            'BestCost': self.BestCost,
            'iteration': iteration,
            'start_time': time.time(),
            'noChangeCount': self.noChangeCount
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="cpso_checkpoint.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.positions = checkpoint['positions']
        self.velocities = checkpoint['velocities']
        self.costs = checkpoint['costs']
        self.best_positions = checkpoint['best_positions']
        self.best_costs = checkpoint['best_costs']
        self.global_best_position = checkpoint['global_best_position']
        self.global_best_cost = checkpoint['global_best_cost']
        self.BestCost = checkpoint['BestCost']
        self.restored_iteration = checkpoint['iteration']
        self.start_time = checkpoint.get('start_time', time.time())
        self.noChangeCount = checkpoint.get('noChangeCount', 0)

