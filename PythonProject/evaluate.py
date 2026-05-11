import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast

# Import moduli locali
from dataset import LogoDataset, TripletLogoDataset, FlickrLogosDataset
from utils import build_query_gallery
from models import LogoNet


model_pth="logonet_resnet50_margin04_E5_LR2.5e-05.pth"
logodet_path="databases/LogoDet-3K"
flicker_path="databases/FlickrLogos32"
result_file_path="evaluation_results_04_LR000025.csv"


def set_seed(seed=42):
    """Fissa la casualità per risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics_and_plots(q_embs, q_labels, g_embs, g_labels, dataset_name, ks=[1, 5, 10]):
    """Calcola i 9 parametri di ranking e genera il grafico CMC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_embs_t = torch.from_numpy(q_embs).to(device)
    g_embs_t = torch.from_numpy(g_embs).to(device)

    # Calcolo distanze vettorizzato in GPU
    dists = torch.cdist(q_embs_t, g_embs_t).cpu().numpy()

    num_queries = len(q_labels)
    num_gallery = len(g_labels)
    mAP, mrr = 0, 0
    recall_counts = {k: 0 for k in ks}
    precision_sums = {k: 0 for k in ks}
    cmc_counts = np.zeros(num_gallery)

    for i in range(num_queries):
        rank_indices = np.argsort(dists[i])
        relevant_matches = (g_labels[rank_indices] == q_labels[i])

        # Recall@K e Precision@K
        for k in ks:
            matches_at_k = relevant_matches[:k]
            if np.any(matches_at_k):
                recall_counts[k] += 1
            precision_sums[k] += np.sum(matches_at_k) / k

        # MRR e CMC
        first_hit = np.where(relevant_matches)[0]
        if len(first_hit) > 0:
            idx = first_hit[0]
            mrr += 1.0 / (idx + 1)
            cmc_counts[idx:] += 1

        # mAP
        hits, sum_prec = 0, 0
        for j, match in enumerate(relevant_matches):
            if match:
                hits += 1
                sum_prec += hits / (j + 1)
        if hits > 0:
            mAP += sum_prec / hits

    # Plot CMC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, min(21, num_gallery + 1)), cmc_counts[:20] / num_queries, marker='o', color='blue')
    plt.title(f"CMC Curve - {dataset_name} - 04_000025")
    plt.xlabel("Rank")
    plt.ylabel("Identification Probability")
    plt.grid(True)
    plt.savefig(f"cmc_{dataset_name}_04_000025.png")
    plt.close()

    res = {f'Recall@{k}': recall_counts[k] / num_queries for k in ks}
    res.update({f'Precision@{k}': precision_sums[k] / num_queries for k in ks})
    res.update({'mAP': mAP / num_queries, 'MRR': mrr / num_queries})
    return res


def get_embs_optimized(ds, model, device):
    """Estrattore di feature con Batch Size alta e AMP."""
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    embs, lbls = [], []
    with torch.no_grad():
        for imgs, l in loader:
            if imgs is not None:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    features = model(imgs.to(device))
                embs.append(features.cpu().numpy())
                lbls.extend(l)
    return np.vstack(embs), np.array(lbls)


def run_evaluation():
    set_seed(42)  # Garantisce riproducibilità
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_pth
    #logodet_path = os.path.join(os.getcwd(), logodet_path)
    flickr_path = os.path.join(os.getcwd(), flicker_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = LogoNet().to(device)
    # CORREZIONE: Aggiunto weights_only=True per eliminare il FutureWarning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_data = []

    # 1. EVALUATION LOGODET-3K (Include calcolo Loss matematica)
    if os.path.exists(logodet_path):
        print("🧪 Valutazione LogoDet-3K...")
        test_base = LogoDataset(root_dir=logodet_path, split="test", transform=transform)

        # Calcolo Loss su triplette di test
        triplet_ds = TripletLogoDataset(test_base)
        loader_loss = DataLoader(triplet_ds, batch_size=64, shuffle=False)
        loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
        t_loss = 0
        with torch.no_grad():
            for a, p, n, _ in loader_loss:
                with autocast(device_type=device.type):
                    t_loss += loss_fn(model(a.to(device)), model(p.to(device)), model(n.to(device))).item()

        q, g = build_query_gallery(test_base)
        res = calculate_metrics_and_plots(*get_embs_optimized(q, model, device),
                                          *get_embs_optimized(g, model, device), "LogoDet-3K")
        res['Loss'] = t_loss / len(loader_loss)
        res['Dataset'] = 'LogoDet-3K'
        all_data.append(res)

    # 2. EVALUATION FLICKRLOGOS-32 (Puro Retrieval)
    if os.path.exists(flickr_path):
        print("📷 Valutazione FlickrLogos-32...")
        flickr_ds = FlickrLogosDataset(root_dir=flickr_path, transform=transform)
        if len(flickr_ds) > 0:
            fq, fg = build_query_gallery(flickr_ds)
            f_res = calculate_metrics_and_plots(*get_embs_optimized(fq, model, device),
                                                *get_embs_optimized(fg, model, device), "FlickrLogos-32")
            f_res['Loss'] = np.nan  # N/A per retrieval puro
            f_res['Dataset'] = 'FlickrLogos-32'
            all_data.append(f_res)

    # 3. SALVATAGGIO CSV RECAP (Tutti i 9 parametri)
    if all_data:
        df = pd.DataFrame(all_data)
        cols = ['Dataset', 'Loss', 'mAP', 'MRR',
                'Precision@1', 'Precision@5', 'Precision@10',
                'Recall@1', 'Recall@5', 'Recall@10']
        df[cols].to_csv(result_file_path, index=False)
        print("\n✅ Valutazione completata. Tabella salvata in 'evaluation_results_04_LR000025.csv'")
        print(df[cols].to_string())


if __name__ == "__main__":
    run_evaluation()