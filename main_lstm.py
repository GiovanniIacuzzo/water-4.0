import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader, random_split

from utils.dataset import LTownLeakageDataset
from models.lstm_model import LSTMPredictor
from utils.train import train_model
from utils.evaluate import evaluate_model
from CPSO.ottimizzazione import optimize_with_cpso
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    experiment = Experiment(
        api_key="hH2SAakLm4RU5yJggbSxDAQ6v",
        project_name="WATER",
        workspace="giovanniiacuzzo2",
    )
    experiment.set_name("Test LSTM-CPSO")

    # ====== Device selection ======
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilizzato: {device}")

    # ====== Dataset loading ======
    dataset = LTownLeakageDataset(
        demand_files=["data/2018_SCADA_Demands.csv"],
        flow_files=["data/2018_SCADA_Flows.csv"],
        level_files=["data/2018_SCADA_Levels.csv"],
        pressure_files=["data/2018_SCADA_Pressures.csv"],
        leakage_files=["data/2018_Leakages.csv"],
        seq_len=288,
        task_type="regression"
    )

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # ====== CPSO optimization ======
    input_size = dataset.inputs.shape[1]
    output_size = dataset.targets.shape[1]

    num_layers, hidden_size, lr, dropout = optimize_with_cpso(
        train_loader,
        val_loader,
        input_size,
        output_size,
        experiment=experiment
    )

    # ====== Model configuration ======
    model_config = {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }

    model = LSTMPredictor(**model_config).to(device)
    model.config = model_config

    experiment.log_parameters({
        "batch_size": 64,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": 288,
        "epochs": 50,
        "optimizer": "CPSO",
        "learning_rate": lr
    })

    # ====== Model training ======
    best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=1,
        lr=lr,
        patience=5,
        experiment=experiment
    )

    print("Salvataggio del modello migliore")
    # Salvataggio del modello
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config
    }, "best_lstm_model.pth")

    checkpoint = torch.load("best_lstm_model.pth", map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]

    model = LSTMPredictor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ====== Final evaluation ======
    evaluate_model(model, test_loader, experiment, name_prefix="Test")
    experiment.end()

""" 
Errore da evitare:
[Errore][Isola 2] MPS backend out of memory (MPS allocated: 3.11 GB, other allocations: 5.42 GB, max     island_cpso.py:15
        allowed: 9.07 GB). Tried to allocate 720.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0                  
        to disable upper limit for memory allocations (may cause system failure).  
Per risolvere:
forza il garbage collector e svuota la cache MPS.
ridurre la batch size o il numero di particelle.
    """