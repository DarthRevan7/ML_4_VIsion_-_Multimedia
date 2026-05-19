import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torchvision import transforms

# Importazione rigida dei parametri dal training attivo
from main_start import margin, margin_clean, lr_clean, n_epochs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import moduli locali
from dataset import LogoDataset, TripletLogoDataset, FlickrLogosDataset
from utils import build_query_gallery
from models import LogoNet

# Parametri correnti assegnati a costanti locali
MARGIN_VAL_ORIGINAL = margin
TARGET_MARGIN = margin_clean
TARGET_LR = lr_clean

# DB Paths e File di Output Master
logodet_path = "databases\\LogoDet-3K"
flicker_path = "databases\\FlickrLogos32"
output_csv = "results\\checkpoints_master_summary.csv"


def set_seed(seed=42):
    """Fissa la casualità per risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_model_state(model_path, device):
    """Carica in modo robusto uno state_dict o un checkpoint wrapper."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
    return state


def calculate_metrics_and_plots(q_embs, q_labels, g_embs, g_labels, dataset_name, current_epoch, margin_str, lr_str, ks=[1, 5, 10], kr=[50, 100, 200]):
    """
    Calcola i parametri di ranking e genera il grafico CMC per lo specifico checkpoint.
    """
    q_embs_t = torch.from_numpy(q_embs)
    g_embs_t = torch.from_numpy(g_embs)
    dists = torch.cdist(q_embs_t, g_embs_t).numpy()

    num_gallery = len(g_labels)
    mAP, mrr = 0.0, 0.0
    recall_sums = {k: 0.0 for k in kr}
    precision_sums = {k: 0.0 for k in ks}
    cmc_counts = np.zeros(num_gallery)
    valid_queries = 0

    for i in range(len(q_labels)):
        rank_indices = np.argsort(dists[i])
        relevant_matches = (g_labels[rank_indices] == q_labels[i])
        total_relevant = int(np.sum(relevant_matches))
        
        if total_relevant == 0:
            continue
        valid_queries += 1

        for k in kr:
            recall_sums[k] += np.sum(relevant_matches[:k]) / total_relevant

        for k in ks:
            precision_sums[k] += np.sum(relevant_matches[:k]) / k

        first_hit = np.where(relevant_matches)[0]
        if len(first_hit) > 0:
            idx = first_hit[0]
            mrr += 1.0 / (idx + 1)
            cmc_counts[idx:] += 1

        hits, sum_prec = 0, 0.0
        for j, match in enumerate(relevant_matches):
            if match:
                hits += 1
                sum_prec += hits / (j + 1)
        mAP += sum_prec / total_relevant

    if valid_queries == 0:
        nan = float('nan')
        res = {f'Recall@{k}': nan for k in kr}
        res.update({f'Precision@{k}': nan for k in ks})
        res.update({'mAP': nan, 'MRR': nan})
        res['Queries_Used'] = 0
        res['Queries_Total'] = len(q_labels)
        return res

    # Plot CMC Curve specifico per questo specifico checkpoint dell'esperimento corrente
    os.makedirs("results", exist_ok=True)
    cmc_len = min(20, num_gallery)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, cmc_len + 1), cmc_counts[:cmc_len] / valid_queries, marker='o', color='blue')
    plt.title(f"CMC Curve - {dataset_name} - E{current_epoch} - M{margin_str}_LR{lr_str}")
    plt.xlabel("Rank")
    plt.ylabel("Identification Probability")
    plt.grid(True)
    plt.savefig(f"results\\cmc_{dataset_name}_E{current_epoch}_M{margin_str}_LR{lr_str}.png")
    plt.close()

    res = {f'Recall@{k}': recall_sums[k] / valid_queries for k in kr}
    res.update({f'Precision@{k}': precision_sums[k] / valid_queries for k in ks})
    res.update({'mAP': mAP / valid_queries, 'MRR': mrr / valid_queries})
    res['Queries_Used'] = valid_queries
    res['Queries_Total'] = len(q_labels)
    return res


def get_embs_optimized(ds, model, device):
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    embs, lbls = [], []
    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for imgs, l in loader:
            if imgs is None:
                continue
            imgs = imgs.to(device)
            if use_cuda:
                from torch.amp import autocast
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


def compute_triplet_loss(triplet_ds, model, device, margin_val):
    loader = DataLoader(triplet_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    loss_fn = nn.TripletMarginLoss(margin=margin_val, p=2)
    use_cuda = torch.cuda.is_available()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for a, p, n, _ in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            batch_size = a.size(0)
            if use_cuda:
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    loss = loss_fn(model(a), model(p), model(n))
            else:
                loss = loss_fn(model(a), model(p), model(n))

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float('nan')


def run_evaluation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # Ordine colonne richiesto esplicitamente per il CSV master
    cols = ['Dataset', 'Epoche', 'Margin', 'LR', 'Loss', 'mAP', 'MRR',
            'Precision@1', 'Precision@5', 'Precision@10',
            'Recall@50', 'Recall@100', 'Recall@200',
            'Queries_Total', 'Queries_Used']

    print("\n" + "="*60)
    print("🎯 PARAMETRI ACCETTATI DA MAIN_START:")
    print(f"   🔹 Margin di addestramento: {TARGET_MARGIN}")
    print(f"   🔹 Learning Rate di addestramento: {TARGET_LR}")
    print(f"   🔹 Epoche totali impostate: {n_epochs}")
    print("="*60 + "\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ----------------------------------------------------------------------
    # PHASE 1: IDENTIFICAZIONE RAGIONATA DEI CHECKPOINT FILTRATI
    # ----------------------------------------------------------------------
    checkpoint_dir = "checkpoints"
    checkpoint_list = []
    
    # Costruiamo la regex inserendo in modo dinamico i parametri di main_start
    # Cattura solo i file strutturati come: checkpoint_epoch_X_LR{TARGET_LR}_M{TARGET_MARGIN}.pth
    escaped_lr = re.escape(TARGET_LR)
    escaped_margin = re.escape(TARGET_MARGIN)
    pattern = re.compile(rf"^checkpoint_epoch_(\d+)_LR{escaped_lr}_M{escaped_margin}\.pth$")
    
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            match = pattern.match(filename)
            if match:
                epoch_num = int(match.group(1))
                full_path = os.path.join(checkpoint_dir, filename)
                checkpoint_list.append((epoch_num, TARGET_LR, TARGET_MARGIN, full_path))

    if not checkpoint_list:
        print(f"❌ Nessun checkpoint trovato per la combinazione attiva (LR: {TARGET_LR} | Margin: {TARGET_MARGIN})")
        return

    # Ordina sequenzialmente da epoca 1 a N
    checkpoint_list.sort(key=lambda x: x[0])
    print(f"🔍 Rilevati {len(checkpoint_list)} checkpoint corrispondenti all'addestramento corrente.\n")

    # ----------------------------------------------------------------------
    # PHASE 2: VERIFICA INCREMENTALE SUL MASTER CSV ESISTENTE
    # ----------------------------------------------------------------------
    already_evaluated = set()
    if os.path.exists(output_csv):
        try:
            df_old = pd.read_csv(output_csv)
            # Memorizza le righe vecchie per evitare doppioni
            for _, row in df_old.iterrows():
                already_evaluated.add((row['Dataset'], int(row['Epoche']), str(row['Margin']), str(row['LR'])))
        except Exception:
            print("⚠️ Errore di lettura sul CSV storico. Verrà ricreato da zero.")

    # ----------------------------------------------------------------------
    # PHASE 3: CARICAMENTO DEI DATASET SE ALMENO UN FILE DEVE RUNNARE
    # ----------------------------------------------------------------------
    # Controlliamo se c'è almeno un calcolo da fare prima di allocare la RAM per i dataset
    need_logodet = False
    need_flickr = False
    
    for epoch_num, lr_str, margin_str, _ in checkpoint_list:
        if ('LogoDet-3K', epoch_num, margin_str, lr_str) not in already_evaluated:
            need_logodet = True
        if ('FlickrLogos-32', epoch_num, margin_str, lr_str) not in already_evaluated:
            need_flickr = True

    logodet_eval_data = None
    if need_logodet and os.path.exists(logodet_path):
        print("📦 [RAM] Caricamento LogoDet-3K per i nuovi checkpoint...")
        test_base = LogoDataset(root_dir=logodet_path, split="test", transform=transform)
        triplet_ds = TripletLogoDataset(test_base, deterministic=True)
        q, g, info = build_query_gallery(test_base)
        logodet_eval_data = (test_base, triplet_ds, q, g, info)

    flickr_eval_data = None
    if need_flickr and os.path.exists(flicker_path):
        print("📦 [RAM] Caricamento FlickrLogos-32 per i nuovi checkpoint...")
        flickr_ds = FlickrLogosDataset(root_dir=flicker_path, transform=transform)
        if len(flickr_ds) > 0:
            fq, fg, f_info = build_query_gallery(flickr_ds)
            flickr_eval_data = (flickr_ds, fq, fg, f_info)

    # ----------------------------------------------------------------------
    # PHASE 4: LOOP DI VALUTAZIONE E AGGIORNAMENTO LIVE CSV
    # ----------------------------------------------------------------------
    model = LogoNet().to(device)

    for epoch_num, lr_str, margin_str, ckpt_path in checkpoint_list:
        
        # Sottoprocesso 1: LogoDet-3K
        key_logodet = ('LogoDet-3K', epoch_num, margin_str, lr_str)
        if key_logodet in already_evaluated:
            print(f"ℹ️ [CSV PRESENTE] LogoDet-3K per Epoca {epoch_num} (M:{margin_str} | LR:{lr_str}) caricato dalla cronologia.")
        elif logodet_eval_data is not None:
            print("\n" + "="*60)
            print("🧪 RUN EVALUATION: LogoDet-3K")
            print(f"   Epoche: {epoch_num}  |  Margin: {margin_str}  |  LR: {lr_str}")
            print("="*60)

            state = load_model_state(ckpt_path, device)
            model.load_state_dict(state)
            model.eval()

            test_base, triplet_ds, q, g, info = logodet_eval_data
            t_loss = compute_triplet_loss(triplet_ds, model, device, MARGIN_VAL_ORIGINAL)
            
            res = calculate_metrics_and_plots(
                *get_embs_optimized(q, model, device),
                *get_embs_optimized(g, model, device),
                "LogoDet-3K",
                current_epoch=epoch_num, margin_str=margin_str, lr_str=lr_str
            )
            res['Dataset'] = 'LogoDet-3K'
            res['Epoche'] = epoch_num
            res['Margin'] = margin_str
            res['LR'] = lr_str
            res['Loss'] = t_loss
            res['Queries_Total'] = res.get('Queries_Total', len(q) + info.get('excluded_singleton_queries', 0))
            res['Queries_Used'] = res.get('Queries_Used', len(q))

            # Aggiornamento incrementale istantaneo del CSV
            df_step = pd.DataFrame([res])
            file_exists = os.path.isfile(output_csv)
            df_step[cols].to_csv(output_csv, mode='a', index=False, header=not file_exists)

        # Sottoprocesso 2: FlickrLogos-32
        key_flickr = ('FlickrLogos-32', epoch_num, margin_str, lr_str)
        if key_flickr in already_evaluated:
            print(f"ℹ️ [CSV PRESENTE] FlickrLogos-32 per Epoca {epoch_num} (M:{margin_str} | LR:{lr_str}) caricato dalla cronologia.")
        elif flickr_eval_data is not None:
            print("\n" + "="*60)
            print("📷 RUN EVALUATION: FlickrLogos-32")
            print(f"   Epoche: {epoch_num}  |  Margin: {margin_str}  |  LR: {lr_str}")
            print("="*60)

            state = load_model_state(ckpt_path, device)
            model.load_state_dict(state)
            model.eval()

            flickr_ds, fq, fg, f_info = flickr_eval_data
            f_res = calculate_metrics_and_plots(
                *get_embs_optimized(fq, model, device),
                *get_embs_optimized(fg, model, device),
                "FlickrLogos-32",
                current_epoch=epoch_num, margin_str=margin_str, lr_str=lr_str
            )
            f_res['Dataset'] = 'FlickrLogos-32'
            f_res['Epoche'] = epoch_num
            f_res['Margin'] = margin_str
            f_res['LR'] = lr_str
            f_res['Loss'] = np.nan
            f_res['Queries_Total'] = f_res.get('Queries_Total', len(fq))
            f_res['Queries_Used'] = f_res.get('Queries_Used', len(fq))

            # Aggiornamento incrementale istantaneo del CSV
            df_step = pd.DataFrame([f_res])
            file_exists = os.path.isfile(output_csv)
            df_step[cols].to_csv(output_csv, mode='a', index=False, header=not file_exists)

    # ----------------------------------------------------------------------
    # PHASE 5: RECAP COMPATTO AD OGNI FINE COMPLETAMENTO RUN
    # ----------------------------------------------------------------------
    if os.path.exists(output_csv):
        print("\n📊 " + " TABELLA RIASSUNTIVA MODELLO ATTIVO " + " 📊")
        df_master = pd.read_csv(output_csv)
        
        # Filtriamo il tabellone a schermo mostrando solo l'addestramento odierno per pulizia
        df_filtered = df_master[(df_master['Margin'].astype(str) == str(TARGET_MARGIN)) & 
                                (df_master['LR'].astype(str) == str(TARGET_LR))]
        
        for d_name in df_filtered['Dataset'].unique():
            print(f"\n📈 TREND DI CRESCITA - DATASET: {d_name.upper()}")
            sub = df_filtered[df_filtered['Dataset'] == d_name].sort_values(by='Epoche')
            print(sub[['Dataset', 'Epoche', 'Margin', 'LR', 'Loss', 'mAP', 'MRR', 'Precision@1']].to_string(index=False))
            
        print(f"\n✅ Master summary salvato ed aggiornato in: {output_csv}")


if __name__ == "__main__":
    run_evaluation()