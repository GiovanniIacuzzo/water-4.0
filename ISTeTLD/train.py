import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LeakScenarioDataset
from models import build_models
import os
import numpy as np

def train():
    print("🚀 Avvio training CGAN...")

    # === Hyperparametri ===
    EPOCHS = 100
    BATCH_SIZE = 64
    NOISE_DIM = 16
    LR = 2e-4
    PATIENCE = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR = "checkpoints"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # === Dataset ===
    train_dataset = LeakScenarioDataset(csv_file='data/leak_data.csv')  # <-- Personalizza path
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # === Modelli ===
    G, D, cond_embedder = build_models(NOISE_DIM)
    G, D, cond_embedder = G.to(DEVICE), D.to(DEVICE), cond_embedder.to(DEVICE)

    # === Ottimizzatori ===
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # === Loss ===
    criterion = nn.BCELoss()

    # === Early stopping ===
    best_loss = float('inf')
    epochs_no_improve = 0

    # === Training loop ===
    for epoch in range(EPOCHS):
        G.train()
        D.train()

        loss_G_epoch = 0
        loss_D_epoch = 0

        for batch in train_loader:
            duration = batch['duration'].to(DEVICE)
            severity = batch['severity'].to(DEVICE)
            real_data = torch.cat([duration, severity], dim=1).float()

            weekday = batch['weekday'].to(DEVICE)
            month = batch['month'].to(DEVICE)
            leak_type = batch['leak_type'].to(DEVICE)
            start_time = batch['start_time'].to(DEVICE)

            cond_vec = cond_embedder(weekday, month, leak_type, start_time)

            # === Train Discriminator ===
            D.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1).to(DEVICE)
            fake_labels = torch.zeros(real_data.size(0), 1).to(DEVICE)

            # Reale
            out_real = D(real_data, cond_vec)
            loss_D_real = criterion(out_real, real_labels)

            # Falso
            z = torch.randn(real_data.size(0), NOISE_DIM).to(DEVICE)
            fake_data = G(z, cond_vec)
            out_fake = D(fake_data.detach(), cond_vec)
            loss_D_fake = criterion(out_fake, fake_labels)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # === Train Generator ===
            G.zero_grad()
            z = torch.randn(real_data.size(0), NOISE_DIM).to(DEVICE)
            fake_data = G(z, cond_vec)
            out_fake = D(fake_data, cond_vec)
            loss_G = criterion(out_fake, real_labels)  # Vuole ingannare D
            loss_G.backward()
            opt_G.step()

            loss_D_epoch += loss_D.item()
            loss_G_epoch += loss_G.item()

        avg_loss_D = loss_D_epoch / len(train_loader)
        avg_loss_G = loss_G_epoch / len(train_loader)

        print(f"[{epoch+1:03d}/{EPOCHS}] 🧠 Loss_D: {avg_loss_D:.4f} | 🎨 Loss_G: {avg_loss_G:.4f}")

        # === Early stopping ===
        if avg_loss_D < best_loss:
            best_loss = avg_loss_D
            epochs_no_improve = 0
            torch.save(G.state_dict(), os.path.join(MODEL_DIR, "best_generator.pth"))
            torch.save(D.state_dict(), os.path.join(MODEL_DIR, "best_discriminator.pth"))
            print("💾 Salvataggio modello (Discriminator migliorato)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("⏹️ Early stopping triggered")
                break

    print("✅ Fine training CGAN.")
