import os
import glob
import random
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from utils import crop_logo

class LogoDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, split_ratio=0.8):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths, self.bboxes, self.labels = [], [], []

        categorie = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_brands = []
        for cat in categorie:
            cat_path = os.path.join(root_dir, cat)
            brands = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            for b in brands: all_brands.append((cat, b))

        all_brands.sort()
        random.seed(42)
        random.shuffle(all_brands)

        split_idx = int(len(all_brands) * split_ratio)
        selected_brands = all_brands[:split_idx] if split == "train" else all_brands[split_idx:]

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
                                self.image_paths.append(img_path)
                                self.bboxes.append((xmin, ymin, xmax, ymax))
                                self.labels.append(brand)
                                break
                except: continue

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = crop_logo(self.image_paths[idx], self.bboxes[idx])
        if self.transform and img:
            img = self.transform(img)
        return img, self.labels[idx]

class TripletLogoDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = {l: [] for l in set(base_dataset.labels)}
        for idx, label in enumerate(base_dataset.labels):
            self.label_to_indices[label].append(idx)
        self.labels_list = list(self.label_to_indices.keys())

    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        anchor_img, anchor_label = self.base_dataset[idx]
        pos_idx = random.choice([i for i in self.label_to_indices[anchor_label] if i != idx]) if len(self.label_to_indices[anchor_label]) > 1 else idx
        positive_img, _ = self.base_dataset[pos_idx]
        neg_label = random.choice([l for l in self.labels_list if l != anchor_label])
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative_img, _ = self.base_dataset[neg_idx]
        return anchor_img, positive_img, negative_img, anchor_label


class FlickrLogosDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset per FlickrLogos-32.
        Struttura attesa: root_dir / brand_name / immagini.jpg
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Esplora le cartelle dei brand (es. 'adidas', 'apple', etc.)
        if os.path.exists(root_dir):
            brands = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            for brand in brands:
                brand_path = os.path.join(root_dir, brand)
                # Cerca immagini con estensioni comuni
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in glob.glob(os.path.join(brand_path, ext)):
                        self.image_paths.append(img_path)
                        self.labels.append(brand)
        else:
            print(f"⚠️ Attenzione: Percorso {root_dir} non trovato.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Per FlickrLogos-32 carichiamo l'immagine intera (o la trasformiamo)
        # Nota: FlickrLogos spesso non ha XML di bounding box nello stesso formato di LogoDet
        from PIL import Image
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"❌ Errore caricamento {self.image_paths[idx]}: {e}")
            return None, None