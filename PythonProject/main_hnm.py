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
logodet_path = r"C:\Users\flavi\OneDrive\Desktop\ML_4_VIsion_-_Multimedia\PythonProject\LogoDet-3K"

# Ripresa dal miglior checkpoint con augmentation
checkpoint_path = "logonet_resnet50_margin04_E15_LR_decay_aug.pth"
start_epoch     = 0       # si riparte da zero con hard mining
n_epochs        = 15      # 10 epoche a LR alto + 5 a LR ridotto

# LR decay
learning_rate   = 0.000025
learning_rate2  = 0.00001
lr_decay_epoch  = 10      # da questa epoca (0-based) si applica LR ridotto

# Frequenza stampe nel terminale
stampa_ogni_n_batch = 100

# Hyperparameters
margin       = 0.4
p            = 2
weight_decay = 0.001
batch_size   = 24
num_workers  = 12

# Nomenclatura File
save_name     = "logonet_resnet50_margin04_E15_LR_decay_aug_hardmining.pth"
save_name_csv = "training_log_margin04_E15_LR_decay_aug_hardmining.csv"


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
    print(f"🚀 Training con Hard Negative Mining iniziato alle: {datetime.now().strftime('%H:%M:%S')}")
    print(f"💻 Device: {device} | Worker: {num_workers} | Batch: {batch_size}")
    print(f"📂 Checkpoint di partenza: {checkpoint_path}")
    print(f"📉 LR: {learning_rate} → {learning_rate2} (decay a epoca {lr_decay_epoch + 1})")

    # --- TRANSFORMS ---
    # Training: con augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Valutazione e mining: senza augmentation — deterministico
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- DATASET ---
    dataset_path = logodet_path
    train_base = LogoDataset(root_dir=dataset_path, split="train", transform=transform_train)
    val_base   = LogoDataset(root_dir=dataset_path, split="test",  transform=transform_val)

    # Hard mining attivo sul training, val rimane deterministico
    train_ds = TripletLogoDataset(train_base, deterministic=False, mining='hard')
    val_ds   = TripletLogoDataset(val_base,   deterministic=True,  mining='random')

    # NOTA: il train_loader viene ricreato dopo ogni on_epoch_start perché
    # le triplette cambiano ad ogni epoca con il mining
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"✅ Dataset caricati — train: {len(train_base)} | val: {len(val_base)}")

    # --- MODELLO ---
    model = LogoNet().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"♻️  Pesi ripristinati da: {checkpoint_path}")

    optimizer     = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    scaler        = GradScaler() if device.type == "cuda" else None

    train_history = []

    # --- LOOP TRAINING ---
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # LR decay manuale
        if epoch == lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate2
            print(f"\n📉 LR ridotto a {learning_rate2} (epoca {epoch + 1})")

        print(f"\n--- Epoca {epoch + 1}/{n_epochs} | LR: {optimizer.param_groups[0]['lr']} | Start: {datetime.now().strftime('%H:%M:%S')} ---")

        # Hard mining: calcola embedding e costruisce le triplette difficili
        # transform_val passato per calcolare embedding senza augmentation
        train_ds.on_epoch_start(
            epoch=epoch,
            model=model,
            device=device,
            transform_val=transform_val
        )

        # Ricrea il DataLoader ad ogni epoca perché le triplette sono cambiate
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)

        # 1. Training
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)

        # 2. Checkpoint
        ckpt_name = f"checkpoints/checkpoint_epoch_{epoch+1}_of_{n_epochs}_hardmining.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"💾 Checkpoint salvato: {ckpt_name}")

        # 3. Validazione
        val_stats = validate_light(model, val_loader, loss_function, device)

        # 4. Logging
        duration = (time.time() - epoch_start) / 60
        log_data = {
            "Epoch":        epoch + 1,
            "LR":           optimizer.param_groups[0]['lr'],
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
    print(f"\n✅ Training completato in {total_time:.2f} ore.")
    print(f"📦 Modello finale: {save_name}")
    print(f"⚠️  Crop falliti — train: {train_base.failed_crop_count} | val: {val_base.failed_crop_count}")

    for base, label in [(train_base, 'train'), (val_base, 'val')]:
        if base.failed_crop_count > 0 and len(base.failed_crop_paths) > 0:
            try:
                pd.DataFrame({'failed_path': base.failed_crop_paths}).to_csv(
                    f'failed_crops_{label}_hardmining.csv', index=False)
                print(f"📝 Report crop falliti: failed_crops_{label}_hardmining.csv")
            except Exception:
                pass


if __name__ == '__main__':
    main()
