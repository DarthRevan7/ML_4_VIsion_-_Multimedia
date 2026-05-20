import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
logodet_path = "datasets\\LogoDet-3K"

# Ripresa dal checkpoint epoca 15
checkpoint_path = "checkpoints/checkpoint_epoch_15_of_15_aug.pth"
start_epoch     = 0      # indice 0-based: riparte da epoca 16
n_epochs        = 10      # totale epoche (15 già fatte + 5 nuove)

# LR — si continua con il LR ridotto della fase 2
learning_rate = 0.000003

# Frequenza stampe nel terminale
stampa_ogni_n_batch = 100

# Hyperparameters
margin       = 0.5
p            = 2
weight_decay = 0.001
batch_size   = 24
num_workers  = 12

lr_clean = np.format_float_positional(learning_rate).split('.')[1]

# Nomenclatura File — nuovi, nessuna sovrascrittura
save_name     = f"logonet_resnet50_margin0{margin*10}_E{n_epochs}_LR{lr_clean}_decay_aug.pth"
save_name_csv = f"training_log_margin0{margin*10}_E{n_epochs}_LR{lr_clean}_decay_aug.csv"


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
    """Calcolo leggero di Loss e Distanze medie."""
    model.eval()
    val_loss = 0.0
    dist_pos_total = 0.0
    dist_neg_total = 0.0
    count = 0
    use_amp = device.type == "cuda"

    for anchor, positive, negative, _ in val_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            anc_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            loss = loss_function(anc_emb, pos_emb, neg_emb)

        val_loss += loss.item()
        dist_pos_total += F.pairwise_distance(anc_emb, pos_emb, p=2).mean().item()
        dist_neg_total += F.pairwise_distance(anc_emb, neg_emb, p=2).mean().item()
        count += 1

    return {
        "val_loss":      val_loss / len(val_loader),
        "mean_dist_pos": dist_pos_total / count,
        "mean_dist_neg": dist_neg_total / count
    }


def main():
    os.makedirs("checkpoints", exist_ok=True)

    # Riproducibilità
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_total_time = time.time()
    print(f"🚀 Training (continuazione da epoca 15) iniziato alle: {datetime.now().strftime('%H:%M:%S')}")
    print(f"💻 Device: {device} | Worker: {num_workers} | Batch: {batch_size}")
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📉 LR: {learning_rate} | Margin: {margin} | Epoche: {start_epoch+1}–{n_epochs}")

    # --- TRANSFORMS ---
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- DATASET ---
    dataset_path = logodet_path
    train_base = LogoDataset(root_dir=dataset_path, split="train", transform=transform_train)
    val_base   = LogoDataset(root_dir=dataset_path, split="test",  transform=transform_val)

    train_ds = TripletLogoDataset(train_base, deterministic=False)
    val_ds   = TripletLogoDataset(val_base,   deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"✅ Dataset caricati — train: {len(train_base)} | val: {len(val_base)}")

    # --- MODELLO — ripristino da checkpoint ---
    model = LogoNet().to(device)
    if start_epoch > 0:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"♻️  Pesi ripristinati da: {checkpoint_path}")
    else:
        print("⚠️  Nessun checkpoint caricato, parto da zero.")

    optimizer     = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    scaler        = GradScaler() if device.type == "cuda" else None

    train_history = []

    # --- LOOP TRAINING (epoche 16–20) ---
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        print(f"\n--- Epoca {epoch + 1}/{n_epochs} | LR: {learning_rate} | Start: {datetime.now().strftime('%H:%M:%S')} ---")

        if hasattr(train_loader.dataset, 'on_epoch_start'):
            try:
                train_loader.dataset.on_epoch_start(epoch)
            except Exception:
                pass

        # 1. Training
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)

        # 2. Checkpoint
        ckpt_name = f"checkpoints/checkpoint_epoch_{epoch+1}_of_{n_epochs}_aug.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"💾 Checkpoint salvato: {ckpt_name}")

        # 3. Validazione
        val_stats = validate_light(model, val_loader, loss_function, device)

        # 4. Logging
        duration = (time.time() - epoch_start) / 60
        log_data = {
            "Epoch":        epoch + 1,
            "LR":           learning_rate,
            "Train_Loss":   avg_train_loss,
            "Val_Loss":     val_stats["val_loss"],
            "Dist_Pos":     val_stats["mean_dist_pos"],
            "Dist_Neg":     val_stats["mean_dist_neg"],
            "Duration_Min": duration
        }
        train_history.append(log_data)
        pd.DataFrame(train_history).to_csv(save_name_csv, index=False)

        print(f"📊 Fine Epoca {epoch + 1} | Tempo: {duration:.2f} min")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_stats['val_loss']:.4f}")
        print(f"   Dist Pos: {val_stats['mean_dist_pos']:.4f} | Dist Neg: {val_stats['mean_dist_neg']:.4f}")

    # Salvataggio modello finale
    torch.save(model.state_dict(), save_name)
    total_time = (time.time() - start_total_time) / 3600
    print(f"\n✅ Completato in {total_time:.2f} ore.")
    print(f"📦 Modello finale: {save_name}")
    print(f"⚠️  Crop falliti — train: {train_base.failed_crop_count} | val: {val_base.failed_crop_count}")

    for base, label in [(train_base, 'train'), (val_base, 'val')]:
        if base.failed_crop_count > 0 and len(base.failed_crop_paths) > 0:
            try:
                pd.DataFrame({'failed_path': base.failed_crop_paths}).to_csv(
                    f'failed_crops_{label}_aug_cont.csv', index=False)
                print(f"📝 Report crop falliti salvato: failed_crops_{label}_aug_cont.csv")
            except Exception:
                pass


if __name__ == '__main__':
    main()
