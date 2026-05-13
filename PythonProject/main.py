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

# Import dai file locali
from dataset import LogoDataset, TripletLogoDataset
from models import LogoNet

'''
PARAMETRI DI ADDESTRAMENTO & PATHS
'''
logodet_path = r"C:\Users\flavi\OneDrive\Desktop\ML_4_VIsion_-_Multimedia\PythonProject\LogoDet-3K"
n_epochs = 10

# Frequenza stampe nel terminale
stampa_ogni_n_batch = 100

# Hyperparameters
margin = 0.4
p = 2
learning_rate = 0.00005
weight_decay = 0.001
batch_size = 24 
num_workers = 12

# Nomenclatura File
save_name = f"logonet_resnet50_margin04_E{n_epochs}_LR{learning_rate}.pth"
save_name_csv = f"training_log_margin04_E{n_epochs}_LR{learning_rate}.csv"

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
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
    train_base = LogoDataset(root_dir=dataset_path, split="train", transform=transform)
    val_base = LogoDataset(root_dir=dataset_path, split="val", transform=transform)
    
    train_loader = DataLoader(TripletLogoDataset(train_base), batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(TripletLogoDataset(val_base, deterministic=True), batch_size=batch_size, 
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
    scaler = GradScaler('cuda') if device.type == "cuda" else None

    train_history = []

    # --- LOOP TRAINING ---
    for epoch in range(n_epochs): #start_epoch
        epoch_start = time.time()
        print(f"\n--- Epoca {epoch + 1}/{n_epochs} | Start: {datetime.now().strftime('%H:%M:%S')} ---")
        
        # 1. Training
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        
        # 2. Salvataggio Preventivo
        torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}_10.pth")
        print(f"💾 Checkpoint salvato: checkpoints/checkpoint_epoch_{epoch+1}.pth")

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

if __name__ == '__main__':
    main()