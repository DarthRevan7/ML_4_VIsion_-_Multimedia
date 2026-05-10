import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

# Import dai file locali
from dataset import LogoDataset, TripletLogoDataset
from models import LogoNet

from torch.amp import GradScaler, autocast


'''
PARAMETRI DI ADDESTRAMENTO & PATHS
'''

logodet_path="LogoDet-3K"


batch_size=24
num_workers=12
margin=0.4
p=2
learning_rate=0.00005
weight=0.001

# ---  LOOP DI TRAINING (RESUME) ---
start_epoch = 60  # Epoche già fatte
n_epochs_extra = 20
total_epochs = start_epoch + n_epochs_extra
# Caricamento log esistente per non perdere i dati delle prime 40 epoche
log_vecchio = "training_log_margin04_LR0001.csv"
log_nuovo = "training_log_margin04_LR00005.csv"




def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    """Esegue un'epoca di training con Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0

    # Inizializza lo scaler per gestire la precisione dimezzata
    scaler = GradScaler('cuda')

    for batch_idx, (anchor, positive, negative, _) in enumerate(dataloader):
        # Spostiamo i dati sulla GPU
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Reset dei gradienti
        optimizer.zero_grad()

        # Forward Pass in precisione mista (FP16)
        with autocast('cuda'):
            anc_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            loss = loss_function(anc_emb, pos_emb, neg_emb)

        # Scaliamo la loss, eseguiamo il backward e aggiorniamo i pesi
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx:03d} | Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)

def main():
    # --- 1. SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    print(f"🚀 Utilizzando il device: {device}")

    # --- 2. CONFIGURAZIONE PATH E TRASFORMAZIONI ---
    dataset_path = os.path.join(os.getcwd(), logodet_path)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- 3. CARICAMENTO DATASET ---
    print(f"⏳ Inizializzazione LogoDet-3K da: {dataset_path}...")
    try:
        train_base = LogoDataset(root_dir=dataset_path, split="train", transform=train_transform)

        if len(train_base) == 0:
            print("❌ Errore: Il dataset è vuoto. Verifica la cartella LogoDet-3K.")
            return

        triplet_ds = TripletLogoDataset(train_base)

        # Batch size 32 (se hai errori di memoria 'OOM', abbassa a 16)
        train_loader = DataLoader(
            triplet_ds,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"✅ Dataset caricato: {len(train_base)} immagini totali.")

    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return

    # --- 4. INIZIALIZZAZIONE MODELLO, LOSS E OTTIMIZZATORE ---
    print("🧠 Configurazione LogoNet (ResNet50)...")
    model = LogoNet().to(device)

    # Caricamento del checkpoint esistente (Margine 0.4)
    checkpoint_path = "logonet_resnet50_margin04_LR0001.pth"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Ripristino pesi dal file: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("⚠️ Attenzione: Checkpoint non trovato. Il training partirà da zero!")

    # Parametri richiesti: Margin 0.2, LR 0.00025, WD 0.001
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)

    


    if os.path.exists(log_vecchio):
        train_history = pd.read_csv(log_vecchio).to_dict('records')
        print(f"📈 Log caricato: {len(train_history)} epoche trovate.")
    else:
        train_history = []

    print(f"🏁 Ripresa training: Epoca {start_epoch + 1} fino a {total_epochs}...")

    for epoch in range(start_epoch, total_epochs):
        print(f"\n--- Epoca {epoch + 1}/{total_epochs} ---")

        # Esegue l'epoca (usa la tua funzione train_one_epoch esistente)
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)

        # Aggiornamento log
        train_history.append({"Epoch": epoch + 1, "Loss": avg_loss})
        pd.DataFrame(train_history).to_csv(log_nuovo, index=False)

        # Salvataggio checkpoint aggiornato ad ogni epoca (sicurezza)
        torch.save(model.state_dict(), "logonet_resnet50_margin04_LR00005.pth")

        print(f"📊 Fine Epoca {epoch + 1} | Loss Media: {avg_loss:.4f}")

    print(f"\n✅ Training completato! Modello finale: {checkpoint_path}")


    # --- 5. LOOP DI TRAINING ---
    #n_epochs = 20
    #train_history = []  # Aggiunto per salvare la storia della loss
    #print(f"🏁 Inizio training per {n_epochs} epoche...")

    #for epoch in range(n_epochs):
        #print(f"\n--- Epoca {epoch + 1}/{n_epochs} ---")
        #avg_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)

        # Salvataggio dati epoca per il log CSV
        #train_history.append({"Epoch": epoch + 1, "Loss": avg_loss})

        # Salvataggio immediato su file
        #pd.DataFrame(train_history).to_csv("training_log_margin04.csv", index=False)

        #print(f"📊 Fine Epoca {epoch + 1} | Loss Media: {avg_loss:.4f}")



    # --- 6. SALVATAGGIO ---
    save_name = "logonet_resnet50_margin04_LR00005.pth"
    torch.save(model.state_dict(), save_name)
    print(f"\n✅ Modello salvato in: {save_name}")
    print(f"✅ Log di training salvato in: {log_nuovo}")


if __name__ == '__main__':
    main()