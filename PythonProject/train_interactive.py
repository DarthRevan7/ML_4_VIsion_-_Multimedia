import os
import sys
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
import matplotlib.pyplot as plt

# Import dai file locali (assicurati che dataset.py, utils.py e models.py siano nella stessa cartella)
from dataset import LogoDataset, TripletLogoDataset, FlickrLogosDataset
from utils import build_query_gallery
from models import LogoNet

'''
======================================================
1. PARAMETRI DI ADDESTRAMENTO INIZIALI & PATHS
======================================================
'''
logodet_path = r"databases\\LogoDet-3K"
flicker_path = r"databases\\FlickrLogos32"

# Checkpoint di partenza
checkpoint_path = "checkpoints/checkpoint_epoch_11_of_15_aug.pth"
start_epoch     = 11       # 0 = parte da capo. Se > 0 cerca di caricare il checkpoint_path
n_epochs        = 15      # totale epoche previste

# Parametri Hyper
initial_learning_rate = 0.0000006
margin       = 0.5
p            = 2
weight_decay = 0.001
batch_size   = 64
num_workers  = 12          # Abbassato a 4 o 8 per evitare blocchi in lettura su Windows, alzalo se hai CPU forte

'''
======================================================
2. FUNZIONI DI FORMATTAZIONE (Nomi Puliti)
======================================================
'''
def get_margin_clean(m):
    """Converte 0.5 in '05', 0.6 in '06' ecc."""
    return f"0{int(m * 10)}"

def get_lr_clean(lr):
    """Converte 0.000003 in '000003' rimuovendo lo '0.' iniziale."""
    lr_str = np.format_float_positional(lr).rstrip('0')
    if '.' in lr_str:
        return lr_str.split('.')[1]
    return lr_str

'''
======================================================
3. FUNZIONI DI TRAINING E VALIDAZIONE LEGGERA
======================================================
'''
def train_one_epoch(model, dataloader, optimizer, loss_function, device, scaler, stampa_ogni=100):
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
        if batch_idx % stampa_ogni == 0:
            print(f"    Batch {batch_idx:03d}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_light(model, val_loader, loss_function, device):
    model.eval()
    val_loss, dist_pos_total, dist_neg_total = 0.0, 0.0, 0.0
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

'''
======================================================
4. FUNZIONI DI EVALUATION (Full Evaluation in RAM)
======================================================
'''
def get_embs_optimized(ds, model, device):
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)
    embs, lbls = [], []
    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for imgs, l in loader:
            if imgs is None: continue
            imgs = imgs.to(device)
            if use_cuda:
                with autocast(device_type="cuda"):
                    features = model(imgs)
            else:
                features = model(imgs)

            embs.append(features.cpu().numpy())
            if isinstance(l, torch.Tensor):
                lbls.extend(l.cpu().numpy())
            else:
                lbls.extend(l)
    return np.vstack(embs), np.array(lbls)


def calculate_metrics(q_embs, q_labels, g_embs, g_labels, ks=[1, 5, 10], kr=[1, 5, 10, 50, 100, 200]):
    q_embs_t, g_embs_t = torch.from_numpy(q_embs), torch.from_numpy(g_embs)
    dists = torch.cdist(q_embs_t, g_embs_t).numpy()

    num_gallery = len(g_labels)
    mAP, mrr = 0.0, 0.0
    recall_sums = {k: 0.0 for k in kr}
    precision_sums = {k: 0.0 for k in ks}
    valid_queries = 0

    for i in range(len(q_labels)):
        rank_indices = np.argsort(dists[i])
        relevant_matches = (g_labels[rank_indices] == q_labels[i])
        total_relevant = int(np.sum(relevant_matches))
        
        if total_relevant == 0: continue
        valid_queries += 1

        for k in kr: recall_sums[k] += np.sum(relevant_matches[:k]) / total_relevant
        for k in ks: precision_sums[k] += np.sum(relevant_matches[:k]) / k

        first_hit = np.where(relevant_matches)[0]
        if len(first_hit) > 0:
            mrr += 1.0 / (first_hit[0] + 1)

        hits, sum_prec = 0, 0.0
        for j, match in enumerate(relevant_matches):
            if match:
                hits += 1
                sum_prec += hits / (j + 1)
        mAP += sum_prec / total_relevant

    # Formattazione risultati sicura in caso di query vuote
    if valid_queries == 0:
        res = {"mAP": float('nan'), "MRR": float('nan')}
        for k in ks: res[f"P@{k}"] = float('nan')
        for k in kr: res[f"R@{k}"] = float('nan')
        return res

    res = {
        "mAP": mAP / valid_queries,
        "MRR": mrr / valid_queries
    }
    for k in ks: res[f"P@{k}"] = precision_sums[k] / valid_queries
    for k in kr: res[f"R@{k}"] = recall_sums[k] / valid_queries
    
    return res

'''
======================================================
5. MAIN LOOP: ADDESTRAMENTO CON PAUSA INTERATTIVA
======================================================
'''
def main():
    # Creazione cartelle necessarie
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)

    # Nomi File Dinamici Iniziali
    margin_clean_init = get_margin_clean(margin)
    lr_clean_init = get_lr_clean(initial_learning_rate)
    
    train_log_file = f"training_logs/training_log_E{start_epoch}-E{n_epochs}_M{margin_clean_init}_LR{lr_clean_init}.csv"
    eval_log_file = f"results/eval_E{start_epoch}-E{n_epochs}_M{margin_clean_init}_LR{lr_clean_init}.csv"

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

    # Gestione Datasets (Training)
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

    print("Caricamento Dataset di Training/Validation (LogoDet-3K)...")
    train_base = LogoDataset(root_dir=logodet_path, split="train", transform=transform_train)
    val_base   = LogoDataset(root_dir=logodet_path, split="test",  transform=transform_val)
    train_ds   = TripletLogoDataset(train_base, deterministic=False)
    val_ds     = TripletLogoDataset(val_base,   deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Inizializzazione Modello
    model = LogoNet().to(device)
    if start_epoch > 0:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"♻️ Pesi ripristinati da: {checkpoint_path}")
    else:
        print("⚠️ Partenza da zero (start_epoch=0).")

    # Inizializzazione variabili interattive
    current_lr = initial_learning_rate
    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    scaler = GradScaler() if device.type == "cuda" else None

    train_history = []
    eval_history = []
    
    # Variabili per memorizzare i dataset di Eval in RAM (Lazy Loading)
    eval_cache = {"logodet_loaded": False, "flickr_loaded": False}

    start_total_time = time.time()
    print(f"\n🚀 Training Iniziato alle: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📁 Log Addestramento in: {train_log_file}")
    print(f"📁 Log Evaluation in: {eval_log_file}")
    
    # ----------------------------------------------------
    # LOOP DELLE EPOCHE
    # ----------------------------------------------------
    for epoch in range(start_epoch, n_epochs):
        ep_idx = epoch + 1
        epoch_start = time.time()
        
        margin_clean = get_margin_clean(margin)
        lr_clean = get_lr_clean(current_lr)
        
        print(f"\n" + "="*60)
        print(f"🔄 EPOCA {ep_idx}/{n_epochs} | Target: M={margin_clean} / LR={lr_clean}")
        print("="*60)

        if hasattr(train_loader.dataset, 'on_epoch_start'):
            try: train_loader.dataset.on_epoch_start(epoch)
            except Exception: pass

        # 1. Training & Light Validation
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        val_stats = validate_light(model, val_loader, loss_function, device)

        # 2. Salvataggio Checkpoint (Convezione Nomi Richiesta)
        ckpt_name = f"checkpoints/checkpoint_E{ep_idx}_M{margin_clean}_LR{lr_clean}.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"\n💾 Checkpoint Salvato: {ckpt_name}")

        duration = (time.time() - epoch_start) / 60
        print(f"📊 Prestazioni Epoca (T={duration:.2f}m) | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_stats['val_loss']:.4f}")
        print(f"   📏 Distanza media Positivi: {val_stats['mean_dist_pos']:.4f} | Negativi: {val_stats['mean_dist_neg']:.4f}")

        # Logging CSV Light
        log_data = {
            "Epoch": ep_idx, 
            "LR": current_lr, 
            "Margin": margin,
            "Train_Loss": avg_train_loss, 
            "Val_Loss": val_stats["val_loss"],
            "Dist_Pos": val_stats["mean_dist_pos"],
            "Dist_Neg": val_stats["mean_dist_neg"]
        }
        train_history.append(log_data)
        pd.DataFrame(train_history).to_csv(train_log_file, index=False)

        # ----------------------------------------------------
        # 🕹️ PANNELLO DI CONTROLLO INTERATTIVO
        # ----------------------------------------------------
        print("\n" + "⚠️ "*10 + " PAUSA INTERATTIVA " + "⚠️ "*10)
        
        # A) Richiesta EVALUATION
        do_eval = input(f"👉 Vuoi fare l'EVALUATION completa su Flickr/LogoDet ora? (s/n): ").strip().lower()
        if do_eval == 's':
            model.eval()
            
            # Caricamento ritardato Flickr
            if not eval_cache["flickr_loaded"]:
                print("📦 [RAM] Caricamento FlickrLogos-32...")
                if os.path.exists(flicker_path):
                    eval_cache["flickr_ds"] = FlickrLogosDataset(root_dir=flicker_path, transform=transform_val)
                    eval_cache["fq"], eval_cache["fg"], _ = build_query_gallery(eval_cache["flickr_ds"])
                    eval_cache["flickr_loaded"] = True
                else:
                    print(f"❌ Percorso non trovato: {flicker_path}")
            
            # Caricamento ritardato LogoDet Eval
            if not eval_cache["logodet_loaded"]:
                print("📦 [RAM] Estrazione Query/Gallery LogoDet-3K...")
                eval_cache["lq"], eval_cache["lg"], _ = build_query_gallery(val_base)
                eval_cache["logodet_loaded"] = True

            print("\n🔍 Risultati Evaluation:")
            if eval_cache["flickr_loaded"]:
                res_f = calculate_metrics(
                    *get_embs_optimized(eval_cache["fq"], model, device),
                    *get_embs_optimized(eval_cache["fg"], model, device)
                )
                print(f"📷 FlickrLogos | mAP: {res_f['mAP']:.4f} | MRR: {res_f['MRR']:.4f} | P@1: {res_f['P@1']:.4f} | R@100: {res_f['R@100']:.4f}")
                
                # Aggiunge i dati al log Eval (Loss vuota per Flickr)
                res_f_log = {"Epoch": ep_idx, "Dataset": "FlickrLogos-32", "Loss": np.nan, **res_f}
                eval_history.append(res_f_log)
            
            if eval_cache["logodet_loaded"]:
                res_l = calculate_metrics(
                    *get_embs_optimized(eval_cache["lq"], model, device),
                    *get_embs_optimized(eval_cache["lg"], model, device)
                )
                print(f"🧪 LogoDet-3K  | mAP: {res_l['mAP']:.4f} | MRR: {res_l['MRR']:.4f} | P@1: {res_l['P@1']:.4f} | R@100: {res_l['R@100']:.4f}")
                
                # Aggiunge i dati al log Eval (Loss ereditata dalla light validation attuale)
                res_l_log = {"Epoch": ep_idx, "Dataset": "LogoDet-3K", "Loss": val_stats["val_loss"], **res_l}
                eval_history.append(res_l_log)
                
            # Salva su disco storico pesante
            if eval_history:
                pd.DataFrame(eval_history).to_csv(eval_log_file, index=False)
                print(f"💾 Risultati Evaluation salvati su: {eval_log_file}")

        # B) Richiesta MODIFICA LR
        change_lr = input(f"👉 Il LR attuale è {current_lr}. Vuoi modificarlo per l'epoca successiva? (s/n): ").strip().lower()
        if change_lr == 's':
            try:
                new_lr_input = input("   Inserisci il nuovo LR (es. 0.000003 oppure 3e-6): ").strip()
                new_lr = float(new_lr_input)
                # Applica nuovo LR all'ottimizzatore a caldo
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
                print(f"✅ LR aggiornato con successo a {current_lr}")
            except ValueError:
                print(f"❌ Valore non valido. Il LR rimane {current_lr}")
        
        print("Ripresa del training in corso...\n")
        # Fine Pausa Interattiva

    total_time = (time.time() - start_total_time) / 3600
    print(f"\n✅ Training completato in {total_time:.2f} ore.")
    print("Verifica i tuoi file salvati nella cartella 'checkpoints/'.")

if __name__ == '__main__':
    main()