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

# Import dai file locali
from dataset import LogoDataset, TripletLogoDataset
from models import LogoNet

'''
PARAMETRI DI ADDESTRAMENTO & PATHS
'''
logodet_path = "databases\\LogoDet-3K"
n_epochs = 5 

#Stampa
stampa_dopo_n_batch = 100

# Hyperparameters
margin = 0.4
p = 2
learning_rate = 0.000025 
weight_decay = 0.001
batch_size = 24 
num_workers = 16 

# Nomenclatura File
save_name = f"logonet_resnet50_margin04_E{n_epochs}_LR{learning_rate}.pth"
save_name_csv = f"training_log_margin04_E{n_epochs}_LR{learning_rate}.csv"

def train_one_epoch(model, dataloader, optimizer, loss_function, device, scaler):
    """Esegue un'epoca di training con Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0

    for batch_idx, (anchor, positive, negative, _) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            anc_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            loss = loss_function(anc_emb, pos_emb, neg_emb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % stampa_dopo_n_batch == 0:
            print(f"   Batch {batch_idx:03d} | Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, val_loader, loss_function, device):
    """
    Calcola Validation Loss e distanze medie Positivi/Negativi.
    """
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

        # Calcolo distanze medie (Euclidea)
        dist_pos = F.pairwise_distance(anc_emb, pos_emb, p=2).mean()
        dist_neg = F.pairwise_distance(anc_emb, neg_emb, p=2).mean()
        
        dist_pos_total += dist_pos.item()
        dist_neg_total += dist_neg.item()
        count += 1

    return {
        "val_loss": val_loss / len(val_loader),
        "mean_dist_pos": dist_pos_total / count,
        "mean_dist_neg": dist_neg_total / count
    }

@torch.no_grad()
def evaluate_retrieval(model, val_base_dataset, device, k_list=[1, 5, 10]):
    """
    Versione corretta: gestisce dataset con metadati extra e ottimizza la RAM.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    # Usiamo un batch più grande per l'estrazione (più veloce)
    eval_loader = DataLoader(val_base_dataset, batch_size=64, shuffle=False, num_workers=4)

    print("🔍 Estrazione embedding per valutazione...")
    for batch in eval_loader:
        # Peschiamo solo i primi due elementi, ignorando il resto (*_)
        images, labels, *_ = batch 
        
        images = images.to(device)
        embeddings = model(images)
        
        all_embeddings.append(embeddings.cpu())
        
        # Se labels è una lista o tupla (per colpa del DataLoader), prendiamo il primo elemento
        if isinstance(labels, (list, tuple)):
            all_labels.append(labels[0].cpu())
        else:
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    num_samples = len(all_labels)
    print(f"📊 Calcolo metriche su {num_samples} campioni...")

    # METRICHE
    correct_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    aps = []

    # Per non saturare la RAM, calcoliamo le distanze riga per riga (o a blocchi)
    # Invece di una matrice N x N da 2.5GB, facciamo confronti diretti
    for i in range(num_samples):
        query_emb = all_embeddings[i].unsqueeze(0)
        query_label = all_labels[i]

        # Calcoliamo le distanze della query verso TUTTI gli altri
        dists = torch.norm(all_embeddings - query_emb, p=2, dim=1)
        dists[i] = float('inf') # Escludiamo se stessi

        # Prendiamo i top 10 vicini
        _, indices = torch.topk(dists, k=10, largest=False)
        retrieved_labels = all_labels[indices]

        # Precision@1
        if query_label == retrieved_labels[0]:
            correct_at_1 += 1
        
        # Recall@K
        if query_label in retrieved_labels[:5]:
            recall_at_5 += 1
        if query_label in retrieved_labels[:10]:
            recall_at_10 += 1

        # mAP (Calcolo sulla riga corrente)
        relevant_mask = (all_labels == query_label).float()
        relevant_mask[i] = 0 # Escludiamo la query stessa
        
        # Ordiniamo tutti i risultati per la query i-esima per avere AP reale
        # Nota: per dataset giganti questo è lento, ma è il modo corretto
        sorted_indices = torch.argsort(dists)
        sorted_relevant = relevant_mask[sorted_indices]
        
        if sorted_relevant.sum() > 0:
            cumulative_relevant = torch.cumsum(sorted_relevant, dim=0)
            precision_at_k = cumulative_relevant / torch.arange(1, num_samples + 1)
            ap = (precision_at_k * sorted_relevant).sum() / sorted_relevant.sum()
            aps.append(ap.item())

    return {
        "p@1": correct_at_1 / num_samples,
        "r@5": recall_at_5 / num_samples,
        "r@10": recall_at_10 / num_samples,
        "mAP": sum(aps) / len(aps) if aps else 0
    }

def main():
    # --- 1. SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    print(f"🚀 Utilizzando il device: {device}")

    # --- 2. TRASFORMAZIONI ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 3. CARICAMENTO DATASET (80/20 Split) ---
    dataset_path = os.path.join(os.getcwd(), logodet_path)
    print(f"⏳ Inizializzazione LogoDet-3K...")
    
    try:
        # Carichiamo i base dataset
        train_base = LogoDataset(root_dir=dataset_path, split="train", transform=train_transform)
        val_base = LogoDataset(root_dir=dataset_path, split="val", transform=val_transform)

        # Creiamo i triplet dataset
        triplet_train_ds = TripletLogoDataset(train_base)
        triplet_val_ds = TripletLogoDataset(val_base)

        train_loader = DataLoader(triplet_train_ds, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True)
        
        val_loader = DataLoader(triplet_val_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        
        print(f"✅ Dataset caricati. Train: {len(train_base)} | Val: {len(val_base)}")

    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return

    # --- 4. INIZIALIZZAZIONE MODELLO, LOSS E OTTIMIZZATORE ---
    print("🧠 Configurazione LogoNet (ResNet50)...")
    model = LogoNet().to(device)
    loss_function = nn.TripletMarginLoss(margin=margin, p=p)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler('cuda')

    # --- 5. LOOP DI TRAINING & VALIDATION ---
    train_history = []
    print(f"🏁 Inizio training per {n_epochs} epoche...")

    for epoch in range(n_epochs):
        print(f"\n--- Epoca {epoch + 1}/{n_epochs} ---")
        
        # 1. TRAINING (La parte più lunga)
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)

        print(f"Fine Epoca {epoch+1} | Loss Media: {avg_train_loss}")
        
        # 2. SALVATAGGIO IMMEDIATO (Se crasha dopo, almeno i pesi sono salvi!)
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"💾 Checkpoint salvato: checkpoint_epoch_{epoch+1}.pth")

        # 3. VALIDATION LOSS (Leggera)
        val_stats = validate(model, val_loader, loss_function, device)
        
        # 4. EVALUATION (Quella pesante che ha crashato)
        # Se crasha qui, non perdiamo il punto 1 e 2!
        retrieval_stats = evaluate_retrieval(model, val_base, device)

    # --- 6. SALVATAGGIO FINALE ---
    torch.save(model.state_dict(), save_name)
    print(f"\n✅ Training completato. Modello finale: {save_name}")

if __name__ == '__main__':
    main()