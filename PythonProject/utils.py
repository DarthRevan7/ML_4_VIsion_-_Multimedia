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


def build_query_gallery(base_test_dataset, rng_seed=42, ensure_positive_in_gallery=True):
    """Divide il dataset di test in Query e Gallery.

    - If a class has only one sample and ensure_positive_in_gallery is True,
      the sample is placed in the gallery (no singleton query created).
    Returns (query_subset, gallery_subset, info_dict)
    info_dict contains counts and excluded query indices.
    """
    label_to_indices = {}
    for idx, label in enumerate(base_test_dataset.labels):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(rng_seed)
    query_indices, gallery_indices = [], []
    excluded_queries = []
    for label in sorted(label_to_indices.keys()):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        if len(indices) == 1 and ensure_positive_in_gallery:
            # singleton: keep it in gallery only
            gallery_indices.extend(indices)
            excluded_queries.append(indices[0])
            continue
        if len(indices) > 0:
            query_indices.append(indices[0])
        if len(indices) > 1:
            gallery_indices.extend(indices[1:])

    from torch.utils.data import Subset
    info = {
        'total_classes': len(label_to_indices),
        'total_queries_created': len(query_indices),
        'excluded_singleton_queries': len(excluded_queries),
        'excluded_query_indices': excluded_queries,
    }
    return Subset(base_test_dataset, query_indices), Subset(base_test_dataset, gallery_indices), info