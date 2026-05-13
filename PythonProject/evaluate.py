import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torchvision import transforms

# Import moduli locali
from dataset import LogoDataset, TripletLogoDataset, FlickrLogosDataset
from utils import build_query_gallery
from models import LogoNet

# Model & Result paths
model_pth = "logonet_resnet50_margin04_E5_LR5e-05.pth"
result_file_path = "final_eval_margin04_E5_LR5e-05.csv"

# DB Paths
logodet_path = "LogoDet-3K"
flicker_path = "FlickrLogos32"

MARGIN = 0.4


def set_seed(seed=42):
    """Fissa la casualità per risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics_and_plots(q_embs, q_labels, g_embs, g_labels, dataset_name, ks=[1, 5, 10]):
    """
    Calcola i 9 parametri di ranking e genera il grafico CMC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_embs_t = torch.from_numpy(q_embs).to(device)
    g_embs_t = torch.from_numpy(g_embs).to(device)

    # Calcolo distanze vettorizzato in GPU
    dists = torch.cdist(q_embs_t, g_embs_t).cpu().numpy()

    num_gallery = len(g_labels)
    mAP, mrr = 0.0, 0.0
    recall_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    cmc_counts = np.zeros(num_gallery)
    valid_queries = 0  # conta solo query con almeno un rilevante in gallery

    for i in range(len(q_labels)):
        rank_indices = np.argsort(dists[i])
        relevant_matches = (g_labels[rank_indices] == q_labels[i])

        # Totale rilevanti per questa query in tutta la gallery
        total_relevant = int(np.sum(relevant_matches))
        if total_relevant == 0:
            # Query senza rilevanti in gallery: saltata per tutte le metriche
            continue
        valid_queries += 1

        # --- Recall@K  ---
        # Recall@K = # rilevanti nei top-K / totale rilevanti per la query
        for k in ks:
            recall_sums[k] += np.sum(relevant_matches[:k]) / total_relevant

        # --- Precision@K ---
        # Precision@K = # rilevanti nei top-K / K
        for k in ks:
            precision_sums[k] += np.sum(relevant_matches[:k]) / k

        # --- MRR e CMC  ---
        first_hit = np.where(relevant_matches)[0]
        if len(first_hit) > 0:
            idx = first_hit[0]
            mrr += 1.0 / (idx + 1)
            cmc_counts[idx:] += 1

        # --- mAP  ---
        # AP = sum(Precision@j * rel@j) / total_relevant  (standard TREC/ImageNet)
        hits, sum_prec = 0, 0.0
        for j, match in enumerate(relevant_matches):
            if match:
                hits += 1
                sum_prec += hits / (j + 1)
        mAP += sum_prec / total_relevant

    if valid_queries == 0:
        print(f"⚠️  [{dataset_name}] Nessuna query con rilevanti in gallery. Metriche non calcolabili.")
        nan = float('nan')
        res = {f'Recall@{k}': nan for k in ks}
        res.update({f'Precision@{k}': nan for k in ks})
        res.update({'mAP': nan, 'MRR': nan})
        return res

    # Plot CMC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, min(21, num_gallery + 1)), cmc_counts[:20] / valid_queries, marker='o', color='blue')
    plt.title(f"CMC Curve - {dataset_name} - margin04_E5_LR5e-05")
    plt.xlabel("Rank")
    plt.ylabel("Identification Probability")
    plt.grid(True)
    plt.savefig(f"cmc_{dataset_name}_margin04_E5_LR5e-05.png")
    plt.close()

    res = {f'Recall@{k}': recall_sums[k] / valid_queries for k in ks}
    res.update({f'Precision@{k}': precision_sums[k] / valid_queries for k in ks})
    res.update({'mAP': mAP / valid_queries, 'MRR': mrr / valid_queries})
    return res


def get_embs_optimized(ds, model, device):
    """
    Estrattore di feature
    """
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    embs, lbls = [], []
    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for imgs, l in loader:
            if imgs is not None:
                imgs = imgs.to(device)

                if use_cuda:
                    from torch.amp import autocast
                    with autocast(device_type='cuda'):
                        features = model(imgs)
                else:
                    features = model(imgs)

                embs.append(features.cpu().numpy())

                if isinstance(l, torch.Tensor):
                    lbls.extend(l.cpu().numpy())
                else:
                    lbls.extend(l)

    return np.vstack(embs), np.array(lbls)


def compute_triplet_loss(triplet_ds, model, device):
    """
    Calcola la Triplet Loss media sul dataset di test.
    La loss è pesata per dimensione batch per gestire correttamente l'ultimo batch incompleto.
    """
    loader = DataLoader(triplet_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
    use_cuda = torch.cuda.is_available()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for a, p, n, _ in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            batch_size = a.size(0)

            if use_cuda:
                from torch.amp import autocast
                with autocast(device_type='cuda'):
                    loss = loss_fn(model(a), model(p), model(n))
            else:
                loss = loss_fn(model(a), model(p), model(n))

            # accumulo pesato per batch size, non media di medie
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float('nan')


def run_evaluation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = LogoNet().to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
    model.eval()

    all_data = []

    # 1. EVALUATION LOGODET-3K
    if os.path.exists(logodet_path):
        print("🧪 Valutazione LogoDet-3K...")
        test_base = LogoDataset(root_dir=logodet_path, split="test", transform=transform)

        # Calcolo Loss su triplette di test
        triplet_ds = TripletLogoDataset(test_base)
        t_loss = compute_triplet_loss(triplet_ds, model, device)

        # Calcolo metriche di retrieval
        q, g = build_query_gallery(test_base)
        res = calculate_metrics_and_plots(
            *get_embs_optimized(q, model, device),
            *get_embs_optimized(g, model, device),
            "LogoDet-3K"
        )
        res['Loss'] = t_loss
        res['Dataset'] = 'LogoDet-3K'
        all_data.append(res)
    else:
        print(f"⚠️  LogoDet-3K non trovato in: {logodet_path}")

    # 2. EVALUATION FLICKRLOGOS-32 (Puro Retrieval — Loss N/A)
    if os.path.exists(flicker_path):
        print("📷 Valutazione FlickrLogos-32...")
        flickr_ds = FlickrLogosDataset(root_dir=flicker_path, transform=transform)
        if len(flickr_ds) > 0:
            fq, fg = build_query_gallery(flickr_ds)
            f_res = calculate_metrics_and_plots(
                *get_embs_optimized(fq, model, device),
                *get_embs_optimized(fg, model, device),
                "FlickrLogos-32"
            )
            f_res['Loss'] = np.nan  # N/A per retrieval puro
            f_res['Dataset'] = 'FlickrLogos-32'
            all_data.append(f_res)
        else:
            print("⚠️  FlickrLogos-32 dataset vuoto.")
    else:
        print(f"⚠️  FlickrLogos-32 non trovato in: {flicker_path}")

    # 3. SALVATAGGIO CSV RECAP
    if all_data:
        df = pd.DataFrame(all_data)
        cols = ['Dataset', 'Loss', 'mAP', 'MRR',
                'Precision@1', 'Precision@5', 'Precision@10',
                'Recall@1', 'Recall@5', 'Recall@10']
        df[cols].to_csv(result_file_path, index=False)
        print(f"\n✅ Valutazione completata. Tabella salvata in {result_file_path}")
        print(df[cols].to_string())
    else:
        print("\n❌ Nessun dataset valutato. Controlla i path configurati.")


if __name__ == "__main__":
    run_evaluation()
