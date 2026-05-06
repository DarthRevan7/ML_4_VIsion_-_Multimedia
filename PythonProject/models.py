import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LogoNet(nn.Module):
    def __init__(self):
        super(LogoNet, self).__init__()

        # Carichiamo ResNet50 con i pesi pre-addestrati su ImageNet
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        # Sostituiamo il layer finale (fc) per emettere un vettore a 2048 dimensioni
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2048)

    def forward(self, x):
        # Generiamo l'embedding
        embedding = self.backbone(x)
        # Applichiamo la normalizzazione L2 (fondamentale per una Triplet Loss stabile)
        return F.normalize(embedding, p=2, dim=1)