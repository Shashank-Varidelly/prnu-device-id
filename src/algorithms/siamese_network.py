"""
Siamese / Triplet Network for PRNU Embedding
==============================================

Embedding-based device identification using contrastive and triplet losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingEncoder(nn.Module):
    """Shared encoder producing L2-normalised embeddings."""

    def __init__(self, in_channels=1, embedding_dim=128, pretrained=False):
        super().__init__()
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = resnet18(weights=None)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=old_conv.bias is not None,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    backbone.conv1.weight = nn.Parameter(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.projection(self.backbone(x)), p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Siamese network with contrastive loss."""

    def __init__(self, in_channels=1, embedding_dim=128, pretrained=False):
        super().__init__()
        self.encoder = EmbeddingEncoder(in_channels, embedding_dim, pretrained)
        self.embedding_dim = embedding_dim

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)


class ContrastiveLoss(nn.Module):
    """Contrastive loss (Chopra et al., 2005)."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2)
        same = label.float()
        diff = 1.0 - same
        loss = same * 0.5 * distance.pow(2) + diff * 0.5 * F.relu(
            self.margin - distance).pow(2)
        return loss.mean()


class TripletNetwork(nn.Module):
    """Triplet network with triplet margin loss."""

    def __init__(self, in_channels=1, embedding_dim=128, pretrained=False):
        super().__init__()
        self.encoder = EmbeddingEncoder(in_channels, embedding_dim, pretrained)
        self.embedding_dim = embedding_dim

    def forward(self, anchor, positive, negative):
        return self.encoder(anchor), self.encoder(positive), self.encoder(negative)


class TripletLoss(nn.Module):
    """Triplet margin loss."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_siamese_epoch(model, loader, optimizer, device, margin=1.0):
    """Train Siamese model for one epoch."""
    model.train()
    criterion = ContrastiveLoss(margin=margin)
    total_loss, total = 0.0, 0

    for anchor, other, pair_label in tqdm(loader, desc="Siamese training", leave=False):
        anchor, other = anchor.to(device), other.to(device)
        pair_label = pair_label.to(device)
        optimizer.zero_grad()
        emb1, emb2 = model(anchor, other)
        loss = criterion(emb1, emb2, pair_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)

    return {"loss": total_loss / max(total, 1), "num_samples": total}


@torch.no_grad()
def evaluate_siamese(model, loader, device, threshold=0.5):
    """Evaluate Siamese model on pair classification."""
    model.eval()
    tp = fp = tn = fn = 0

    for anchor, other, pair_label in tqdm(loader, desc="Siamese eval", leave=False):
        anchor, other = anchor.to(device), other.to(device)
        emb1, emb2 = model(anchor, other)
        distances = F.pairwise_distance(emb1, emb2)
        predictions = (distances < threshold).long()
        labels = pair_label.to(device)
        tp += ((predictions == 1) & (labels == 1)).sum().item()
        fp += ((predictions == 1) & (labels == 0)).sum().item()
        tn += ((predictions == 0) & (labels == 0)).sum().item()
        fn += ((predictions == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    return {"accuracy": (tp + tn) / max(total, 1),
            "far": fp / max(fp + tn, 1), "frr": fn / max(fn + tp, 1),
            "num_samples": total}


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract embeddings for all samples in a loader."""
    model.eval()
    encoder = model.encoder if hasattr(model, "encoder") else model
    all_emb, all_labels = [], []

    for patches, labels in tqdm(loader, desc="Extracting embeddings", leave=False):
        emb = encoder(patches.to(device))
        all_emb.append(emb.cpu())
        all_labels.extend(labels.tolist() if torch.is_tensor(labels) else labels)

    return torch.cat(all_emb, dim=0), all_labels
