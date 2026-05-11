import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm 

# Import dai tuoi file
from dataset import LogoDataset
from models import LogoNet

'''
CONFIGURAZIONE EVAL
'''
logodet_path = "databases\\LogoDet-3K"
checkpoints_dir = "checkpoints"
results_file = "extra_metrics_log.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_retrieval_pro(model, val_loader_base, device):
    """Calcolo mAP e Precision@1 gestendo etichette stringa o numeriche."""
    model.eval()
    all_embeddings = []
    all_labels = []

    print("   ↳ Estrazione embedding...")
    for batch in val_loader_base:
        images, labels, *_ = batch
        embeddings = model(images.to(device))
        all_embeddings.append(embeddings.cpu())
        
        # Se sono Tensor (numeri), li convertiamo in lista
        if torch.is_tensor(labels):
            all_labels.extend(labels.cpu().tolist())
        else:
            # Se sono stringhe, le aggiungiamo direttamente alla lista
            all_labels.extend(list(labels))

    all_embeddings = torch.cat(all_embeddings)
    num_samples = len(all_labels)

    correct_at_1 = 0
    aps = []

    print(f"   ↳ Calcolo metriche su {num_samples} campioni...")
    for i in tqdm(range(num_samples), desc="      Metriche", leave=False):
        query_emb = all_embeddings[i].unsqueeze(0)
        query_label = all_labels[i]

        # Distanze Euclidee
        dists = torch.norm(all_embeddings - query_emb, p=2, dim=1)
        dists[i] = float('inf') 

        # Precision@1
        _, indices = torch.topk(dists, k=1, largest=False)
        if query_label == all_labels[indices[0]]:
            correct_at_1 += 1
        
        # mAP - Creiamo la maschera di rilevanza (1 se stessa classe, 0 altrimenti)
        rel_mask = torch.tensor([1.0 if l == query_label else 0.0 for l in all_labels])
        rel_mask[i] = 0
        
        if rel_mask.sum() > 0:
            sorted_indices = torch.argsort(dists)
            sorted_relevant = rel_mask[sorted_indices]
            cum_rel = torch.cumsum(sorted_relevant, dim=0)
            prec = cum_rel / torch.arange(1, num_samples + 1)
            ap = (prec * sorted_relevant).sum() / rel_mask.sum()
            aps.append(ap.item())

    return {
        "P@1": correct_at_1 / num_samples,
        "mAP": sum(aps) / len(aps) if aps else 0
    }

def main():
    print(f"🧐 Inizio valutazione extra sui checkpoint in '{checkpoints_dir}'")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_base = LogoDataset(root_dir=os.path.join(os.getcwd(), logodet_path), split="val", transform=transform)
    val_loader = DataLoader(val_base, batch_size=64, shuffle=False, num_workers=8)

    checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")])
    
    if not checkpoint_files:
        print("❌ Nessun checkpoint trovato!")
        return

    all_results = []
    model = LogoNet().to(device)

    for ckpt_name in checkpoint_files:
        epoch_num = ckpt_name.split('_')[-1].replace('.pth', '')
        print(f"\n📂 Analizzando Epoca {epoch_num} ({ckpt_name})...")
        
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"⚠️ Errore caricamento {ckpt_name}: {e}. Salto.")
            continue

        metrics = evaluate_retrieval_pro(model, val_loader, device)
        metrics["Epoch"] = epoch_num
        all_results.append(metrics)
        
        # Salvataggio immediato
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        print(f"✅ Risultati Epoca {epoch_num} salvati! P@1: {metrics['P@1']:.4f} | mAP: {metrics['mAP']:.4f}")

    print(f"\n🏁 Valutazione completata! File: {results_file}")

if __name__ == '__main__':
    main()