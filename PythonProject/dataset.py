import os
import glob
import random
from PIL import Image
import xml.etree.ElementTree as ET
import torch
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

    # FIX: Deve restituire la lunghezza del dataset originale
    def __len__(self):
        return len(self.base_dataset)


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
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.bboxes = []
        self.labels = []

        if not os.path.exists(root_dir):
            print(f"⚠️ Percorso non trovato: {root_dir}")
            return

        # Cerchiamo tutti i file XML ricorsivamente
        xml_pattern = os.path.join(root_dir, "**/*.xml")
        xml_files = glob.glob(xml_pattern, recursive=True)

        for xml_file in xml_files:
            try:
                xml_dir = os.path.dirname(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # 1. Prendiamo il nome del file immagine
                filename_tag = root.find('filename')
                if filename_tag is None or not filename_tag.text:
                    continue
                
                image_filename = filename_tag.text
                img_path = os.path.join(xml_dir, image_filename)

                if not os.path.exists(img_path):
                    continue

                # 2. Prendiamo SOLO IL PRIMO oggetto trovato (UN SOLO logo per immagine)
                obj = root.find('object') 
                if obj is not None:
                    label = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    
                    if bndbox is not None:
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))

                        # Aggiungiamo i dati una sola volta per questo XML
                        self.image_paths.append(img_path)
                        self.bboxes.append((xmin, ymin, xmax, ymax))
                        self.labels.append(label)

            except Exception as e:
                continue

        print(f"✅ FlickrLogos-32 caricato: {len(self.image_paths)} immagini univoche caricate.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = crop_logo(self.image_paths[idx], self.bboxes[idx])
        if img is None:
            return torch.zeros(3, 224, 224), "error"
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]