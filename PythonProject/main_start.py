import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from datetime import datetime
import time
import numpy as np
import random

# Import dai file locali
from dataset import LogoDataset, TripletLogoDataset
from models import LogoNet

'''
PARAMETRI DI ADDESTRAMENTO & PATHS
'''
logodet_path = "databases\\LogoDet-3K"
n_epochs = 5

# Frequenza stampe nel terminale
stampa_ogni_n_batch = 100

# Hyperparameters
margin = 0.5
p = 2
learning_rate = 0.000003
weight_decay = 0.001
batch_size = 64
num_workers = 12

# Nomenclatura File
margin_clean = f"{margin:.1f}".replace('.', '')
lr_clean = np.format_float_positional(learning_rate).split('.')[1]

save_name = f"logonet_resnet50_margin{margin_clean}_E{n_epochs}_LR{lr_clean}.pth"
save_name_csv = f"training_log_margin{margin_clean}_E{n_epochs}_LR{lr_clean}.csv"



def train_one_epoch(model, dataloader, optimizer, loss_function, device, scaler):
    """Esegue il training puro."""
    model.train()
    running_loss = 0.0
    use_amp = device.type == "cuda"
    for batch_idx, (anchor, positive, negative, _) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            anc_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            loss = loss_function(anc_emb, pos_emb, neg_emb)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if batch_idx % stampa_ogni_n_batch == 0:
            print(f"    Batch {batch_idx:03d} | Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)

@torch.no_grad()
def validate_light(model, val_loader, loss_function, device):
    """Calcolo leggero di Loss e Distanze medie (Sicuro per la RAM)."""
    model.eval()
    val_loss = 0.0
    dist_pos_total = 0.0
    dist_neg_total = 0.0
    count = 0
    for anchor, positive, negative, _ in val_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        anc_emb = model(anchor)
        pos_emb = model(positive)
        neg_emb = model(negative)
        
        loss = loss_function(anc_emb, pos_emb, neg_emb)
        val_loss += loss.item()

        # Distanze medie euclidee
        dist_pos_total += F.pairwise_distance(anc_emb, pos_emb, p=2).mean().item()
        dist_neg_total += F.pairwise_distance(anc_emb, neg_emb, p=2).mean().item()
        count += 1

    return {
        "val_loss": val_loss / len(val_loader),
        "mean_dist_pos": dist_pos_total / count,
        "mean_dist_neg": dist_neg_total / count
    }

def main():
    # Crea la cartella checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Salvataggio modello in: {save_name} | Log CSV: {save_name_csv}")
    
    # Configurazione per riproducibilità: attiva se volete run bit-for-bit riproducibili
    reproducible = False # Cambia a True se vuoi massima riproducibilità (può rallentare o crashare)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if reproducible:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    start_total_time = time.time()
    print(f"🚀 Training iniziato alle: {datetime.now().strftime('%H:%M:%S')}")
    print(f"💻 Device: {device} | Worker: {num_workers} | Batch: {batch_size}")

    # Transforms e Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = os.path.join(os.getcwd(), logodet_path)
    print("DEBUG: Caricamento dataset...")
    train_base = LogoDataset(root_dir=dataset_path, split="train", transform=transform)
    val_base = LogoDataset(root_dir=dataset_path, split="val", transform=transform)
    
    # Training: campionamento stocastico delle triplette per aumentare la varietà tra epoche
    train_ds = TripletLogoDataset(train_base, deterministic=False)
    val_ds = TripletLogoDataset(val_base, deterministic=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"✅ Dataset caricati (80/20 split).")

    model = LogoNet().to(device)

    # --- RIPARTENZA DALL'INIZIO DELL'EPOCA 4 ---
    # Carichiamo il lavoro finito dell'epoca 3
    #checkpoint_path = "checkpoints/checkpoint_epoch_3.pth"

    #if os.path.exists(checkpoint_path):
        #print(f"♻️ Ripristino completato fino all'epoca 3. Parto con l'epoca 4...")
        #model.load_state_dict(torch.load(checkpoint_path))
        #start_epoch = 3  # L'indice 3 nel range(0, 5) è la quarta epoca
    #else:
        #print("⚠️ Checkpoint epoca 3 non trovato! Controlla il nome del file.")
        #start_epoch = 0
    # ------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    scaler = GradScaler() if device.type == "cuda" else None

    train_history = []

    # --- LOOP TRAINING ---
    for epoch in range(n_epochs): #start_epoch
        epoch_start = time.time()
        print(f"\n--- Epoca {epoch + 1}/{n_epochs} | Start: {datetime.now().strftime('%H:%M:%S')} ---")
        # Permetti al dataset di training di rigenerare le triplette (se implementato)
        if hasattr(train_loader.dataset, 'on_epoch_start'):
            try:
                train_loader.dataset.on_epoch_start(epoch)
            except Exception:
                pass
        # 1. Training
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        
        # 2. Salvataggio Preventivo
        torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}_LR{lr_clean}_M{margin_clean}.pth")
        print(f"💾 Checkpoint salvato: checkpoints/checkpoint_epoch_{epoch+1}_LR{lr_clean}_M{margin_clean}.pth")

        # 3. Validazione Leggera
        val_stats = validate_light(model, val_loader, loss_function, device)

        # 4. LOGGING IN APPEND (Non sovrascrive il CSV)
        #duration = (time.time() - epoch_start) / 60
        #log_data = {
            #"Epoch": epoch + 1,
            #"Train_Loss": avg_train_loss,
            #"Val_Loss": val_stats["val_loss"],
            #"Dist_Pos": val_stats["mean_dist_pos"],
            #"Dist_Neg": val_stats["mean_dist_neg"],
            #"Duration_Min": duration
        #}

        #df_epoch = pd.DataFrame([log_data])
        #file_exists = os.path.isfile(save_name_csv)
        #df_epoch.to_csv(save_name_csv, mode='a', index=False, header=not file_exists)

        # Logging
        duration = (time.time() - epoch_start) / 60
        log_data = {
            "Epoch": epoch + 1,
            "Train_Loss": avg_train_loss,
            "Val_Loss": val_stats["val_loss"],
            "Dist_Pos": val_stats["mean_dist_pos"],
            "Dist_Neg": val_stats["mean_dist_neg"],
            "Duration_Min": duration
        }
        train_history.append(log_data)
        pd.DataFrame(train_history).to_csv(save_name_csv, index=False)

        print(f"📊 Fine Epoca {epoch + 1} | Tempo: {duration:.2f} min")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_stats['val_loss']:.4f}")
        print(f"   Dist Pos: {val_stats['mean_dist_pos']:.4f} | Dist Neg: {val_stats['mean_dist_neg']:.4f}")

    # Salvataggio finale
    torch.save(model.state_dict(), save_name)
    total_time = (time.time() - start_total_time) / 3600
    print(f"\n✅ Training completato in {total_time:.2f} ore.")
    print(f"📦 Modello finale: {save_name}")
    print(f"⚠️ Crop falliti - train: {train_base.failed_crop_count} | val: {val_base.failed_crop_count}")

    # Salviamo report dei crop falliti per ispezione, se presenti
    if train_base.failed_crop_count > 0 and len(train_base.failed_crop_paths) > 0:
        try:
            pd.DataFrame({'failed_path': train_base.failed_crop_paths}).to_csv('failed_crops_train.csv', index=False)
            print('📝 Report crop falliti salvato: failed_crops_train.csv')
        except Exception:
            pass
    if val_base.failed_crop_count > 0 and len(val_base.failed_crop_paths) > 0:
        try:
            pd.DataFrame({'failed_path': val_base.failed_crop_paths}).to_csv('failed_crops_val.csv', index=False)
            print('📝 Report crop falliti salvato: failed_crops_val.csv')
        except Exception:
            pass

if __name__ == '__main__':
    main()