import os
import random
from PIL import Image
import torch


def crop_logo(image_path, bbox):
    """Ritaglia l'immagine in base alla bounding box (xmin, ymin, xmax, ymax)."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img.crop(bbox)
    except Exception:
        return None


def build_query_gallery(base_test_dataset):
    """Divide il dataset di test in Query e Gallery."""
    label_to_indices = {}
    for idx, label in enumerate(base_test_dataset.labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    rng = random.Random(42)
    query_indices, gallery_indices = [], []
    for label in sorted(label_to_indices.keys()):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        if len(indices) > 0:
            query_indices.append(indices[0])
        if len(indices) > 1:
            gallery_indices.extend(indices[1:])

    from torch.utils.data import Subset
    return Subset(base_test_dataset, query_indices), Subset(base_test_dataset, gallery_indices)