import os
import glob
import random
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import crop_logo


# --- CLASSE SPOSTATA FUORI PER EVITARE ERRORI DI PICKLING ---
class _PlainDataset(Dataset):
    def __init__(self, base, tf):
        self.base = base
        self.tf = tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base._load_raw_image(idx)
        if img is None:
            raise RuntimeError(f"Invalid crop at index {idx}")
        return self.tf(img), idx


class LogoDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, split_ratio=0.8, val_ratio=0.1):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths, self.bboxes, self.labels = [], [], []
        self.failed_crop_count = 0
        self.failed_crop_paths = []

        categorie = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_brands = []
        for cat in categorie:
            cat_path = os.path.join(root_dir, cat)
            brands = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            for b in brands:
                all_brands.append((cat, b))

        all_brands.sort()
        rng = random.Random(42)
        rng.shuffle(all_brands)

        total_brands = len(all_brands)
        train_end = int(total_brands * split_ratio)
        val_end = train_end + int(total_brands * val_ratio)

        if split == "train":
            selected_brands = all_brands[:train_end]
        elif split == "val":
            selected_brands = all_brands[train_end:val_end]
        elif split == "test":
            selected_brands = all_brands[val_end:]
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        for cat, brand in selected_brands:
            brand_path = os.path.join(root_dir, cat, brand)
            xml_files = glob.glob(os.path.join(brand_path, "*.xml"))
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    bndbox = root.find('.//bndbox')
                    if bndbox is not None:
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        base_name = os.path.splitext(os.path.basename(xml_file))[0]
                        for ext in ['.jpg', '.jpeg', '.png']:
                            img_path = os.path.join(brand_path, base_name + ext)
                            if os.path.exists(img_path):
                                try:
                                    img = crop_logo(img_path, (xmin, ymin, xmax, ymax))
                                except Exception:
                                    img = None
                                if img is None:
                                    self.failed_crop_count += 1
                                    if len(self.failed_crop_paths) < 1000:
                                        self.failed_crop_paths.append(img_path)
                                    break
                                self.image_paths.append(img_path)
                                self.bboxes.append((xmin, ymin, xmax, ymax))
                                self.labels.append(brand)
                                break
                except Exception:
                    continue

    def __len__(self):
        return len(self.image_paths)

    def _load_raw_image(self, idx):
        return crop_logo(self.image_paths[idx], self.bboxes[idx])

    def __getitem__(self, idx):
        img = self._load_raw_image(idx)
        if img is None:
            raise RuntimeError(f"Invalid crop at index {idx} for path {self.image_paths[idx]}")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class TripletLogoDataset(Dataset):
    def __init__(self, base_dataset, deterministic=False, seed=42, mining='random'):
        self.base_dataset = base_dataset
        self.deterministic = deterministic
        self.seed = seed
        self.mining = mining

        self.label_to_indices = {l: [] for l in sorted(set(base_dataset.labels))}
        for idx, label in enumerate(base_dataset.labels):
            self.label_to_indices[label].append(idx)
        self.labels_list = list(self.label_to_indices.keys())

        if len(self.labels_list) < 2:
            raise ValueError("TripletLogoDataset requires at least two different classes")

        self.all_indices = list(range(len(base_dataset.labels)))
        self._embeddings = None
        self.triplets = []
        if self.deterministic:
            self._regenerate_triplets(self.seed)

    def update_embeddings(self, model, device, transform):
        model.eval()
        # Uso num_workers=0 per evitare problemi di pickling su Windows con le classi globali
        plain_ds = _PlainDataset(self.base_dataset, transform)
        loader = DataLoader(plain_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        all_embs = []
        with torch.no_grad():
            for imgs, _ in loader:
                embs = model(imgs.to(device))
                all_embs.append(embs.cpu())

        self._embeddings = torch.cat(all_embs, dim=0)  # (N, D)
        model.train()

    def _build_hard_triplets(self):
        """Versione vettorizzata su GPU: riduce i tempi da minuti a secondi."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings non disponibili. Chiama update_embeddings() prima.")

        triplets = []
        embs = self._embeddings
        N = len(embs)
        labels_np = np.array(self.base_dataset.labels)

        # Sposta tutto su GPU una volta sola
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs_gpu = embs.to(device)

        # Batch per il calcolo delle distanze (evita OOM su GPU)
        batch_size = 1000

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_query = embs_gpu[start_idx:end_idx]  # (B, D)

            # Matrice distanze batch: (B, N)
            dists = torch.cdist(batch_query, embs_gpu)

            for i, anchor_idx in enumerate(range(start_idx, end_idx)):
                anchor_label = labels_np[anchor_idx]

                # Positivo (random stessa classe)
                pos_candidates = [idx for idx in self.label_to_indices[anchor_label] if idx != anchor_idx]
                if not pos_candidates:
                    pos_idx, use_synthetic = anchor_idx, True
                else:
                    pos_idx, use_synthetic = random.choice(pos_candidates), False

                # Negativo Hard (classe diversa, distanza minima)
                mask = torch.from_numpy(labels_np == anchor_label).to(device)
                row_dists = dists[i].clone()
                row_dists[mask] = float('inf')

                neg_idx = int(torch.argmin(row_dists).item())
                triplets.append((anchor_idx, pos_idx, neg_idx, anchor_label, use_synthetic))

        self.triplets = triplets

    def _regenerate_triplets(self, seed=None):
        self.triplets = []
        rng = random.Random(self.seed if seed is None else seed)
        for idx in self.all_indices:
            anchor_label = self.base_dataset.labels[idx]
            pos_candidates = [i for i in self.label_to_indices[anchor_label] if i != idx]
            use_synthetic = len(pos_candidates) == 0
            pos_idx = rng.choice(pos_candidates) if pos_candidates else idx
            neg_label = rng.choice([l for l in self.labels_list if l != anchor_label])
            neg_idx = rng.choice(self.label_to_indices[neg_label])
            self.triplets.append((idx, pos_idx, neg_idx, anchor_label, use_synthetic))

    def on_epoch_start(self, epoch=None, model=None, device=None, transform_val=None):
        if self.mining == 'hard':
            print(f"   🔍 Hard mining: calcolo embedding...")
            self.update_embeddings(model, device, transform_val)
            print(f"   🎯 Ricerca negativi difficili su GPU...")
            self._build_hard_triplets()
            print(f"   ✅ Hard mining completato: {len(self.triplets)} triplette.")
        elif self.deterministic:
            seed = (self.seed + epoch) if epoch is not None else self.seed
            self._regenerate_triplets(seed)

    def _make_singleton_positive(self, anchor_img, anchor_idx):
        if isinstance(anchor_img, torch.Tensor):
            return torch.flip(anchor_img, dims=[2])
        return anchor_img

    def __len__(self):
        return len(self.triplets) if (self.mining == 'hard' or self.deterministic) else len(self.all_indices)

    def __getitem__(self, idx):
        if self.mining == 'hard' or self.deterministic:
            anchor_idx, pos_idx, neg_idx, anchor_label, use_synthetic = self.triplets[idx]
        else:
            anchor_idx = self.all_indices[idx]
            anchor_label = self.base_dataset.labels[anchor_idx]
            pos_candidates = [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
            use_synthetic = len(pos_candidates) == 0
            pos_idx = random.choice(pos_candidates) if pos_candidates else anchor_idx
            neg_label = random.choice([l for l in self.labels_list if l != anchor_label])
            neg_idx = random.choice(self.label_to_indices[neg_label])

        anchor_img, _ = self.base_dataset[anchor_idx]
        positive_img = self._make_singleton_positive(anchor_img, anchor_idx) if use_synthetic else \
        self.base_dataset[pos_idx][0]
        negative_img, _ = self.base_dataset[neg_idx]
        return anchor_img, positive_img, negative_img, anchor_label


class FlickrLogosDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.bboxes = []
        self.labels = []
        self.failed_crop_count = 0
        self.failed_crop_paths = []

        if not os.path.exists(root_dir):
            print(f"⚠️ Percorso non trovato: {root_dir}")
            return

        xml_pattern = os.path.join(root_dir, "**/*.xml")
        xml_files = glob.glob(xml_pattern, recursive=True)

        for xml_file in xml_files:
            try:
                xml_dir = os.path.dirname(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()

                filename_tag = root.find('filename')
                if filename_tag is None or not filename_tag.text:
                    continue

                image_filename = filename_tag.text
                img_path = os.path.join(xml_dir, image_filename)

                if not os.path.exists(img_path):
                    continue

                obj = root.find('object')
                if obj is not None:
                    label = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        self.image_paths.append(img_path)
                        self.bboxes.append((xmin, ymin, xmax, ymax))
                        self.labels.append(label)

            except Exception:
                continue

        print(f"✅ FlickrLogos-32 caricato: {len(self.image_paths)} immagini univoche caricate.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = crop_logo(self.image_paths[idx], self.bboxes[idx])
        if img is None:
            self.failed_crop_count += 1
            if len(self.failed_crop_paths) < 10:
                self.failed_crop_paths.append(self.image_paths[idx])
                print(f"⚠️ Crop fallito per: {self.image_paths[idx]}")
            return torch.zeros(3, 224, 224), "error"
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
