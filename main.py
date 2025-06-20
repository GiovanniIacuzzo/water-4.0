from comet_ml import Experiment
from utils.dataset import LTownLeakageDataset
from models.lstm_model import LSTMPredictor
from utils.train import train_model
from utils.evaluate import evaluate_model
from torch.utils.data import DataLoader, random_split
import torch
from CPSO.ottimizzazione import optimize_with_cpso

if __name__ == "__main__":
    experiment = Experiment(
        api_key="hH2SAakLm4RU5yJggbSxDAQ6v",
        project_name="WATER",
        workspace="giovanniiacuzzo2",
    )

    experiment.set_name("Test LSTM-CPSO")

    # Caricamento del dataset
    dataset = LTownLeakageDataset(
        demand_files=["data/2018_SCADA_Demands.csv", "data/2019_SCADA_Demands.csv"],
        flow_files=["data/2018_SCADA_Flows.csv", "data/2019_SCADA_Flows.csv"],
        level_files=["data/2018_SCADA_Levels.csv", "data/2019_SCADA_Levels.csv"],
        pressure_files=["data/2018_SCADA_Pressures.csv", "data/2019_SCADA_Pressures.csv"],
        leakage_files=["data/2018_Leakages.csv", "data/2019_Leakages.csv"],
        seq_len=288,
        task_type="regression"
    )

    # Split del dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Ottimizzazione con CPSO
    input_size = dataset.inputs.shape[1]
    output_size = dataset.targets.shape[1]

    num_layers, hidden_size, lr, dropout = optimize_with_cpso(
        train_loader,
        val_loader,
        input_size,
        output_size,
        experiment=experiment
    )

    # Configurazione del modello con i parametri CPSO
    model_config = {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }

    # Inizializzazione modello
    model = LSTMPredictor(**model_config)
    model.config = model_config

    # Logging dei parametri su Comet
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

    # Addestramento del modello
    train_model(model, train_loader, val_loader, n_epochs=50, experiment=experiment, patience=5, lr=lr)
    model.load_state_dict(torch.load("best_model.pth"))

    # Valutazione finale
    evaluate_model(model, test_loader, experiment, name_prefix="Test")
    experiment.end()