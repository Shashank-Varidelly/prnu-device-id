"""
CNN Classifier — ResNet-18 for PRNU Residual Patches
=====================================================

Classifies PRNU noise residual patches directly into device classes.
Uses a modified ResNet-18 backbone adapted for single-channel input.
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
from pathlib import Path

logger = logging.getLogger(__name__)


class PRNUResNet(nn.Module):
    """ResNet-18 adapted for PRNU residual patch classification.

    Parameters
    ----------
    num_classes : int
        Number of device classes (cameras).
    in_channels : int
        Number of input channels (1 for grayscale, 3 for colour).
    pretrained : bool
        If True, initialise from ImageNet weights.
    dropout : float
        Dropout probability before the final FC layer.
    embedding_dim : int
        Dimension of the penultimate embedding (0 = use ResNet default 512).
    """

    def __init__(self, num_classes, in_channels=1, pretrained=False,
                 dropout=0.3, embedding_dim=0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

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

        if embedding_dim > 0:
            self.embedding = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim), nn.ReLU(inplace=True),
            )
            fc_input = embedding_dim
        else:
            self.embedding = nn.Identity()
            fc_input = feature_dim

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(fc_input, num_classes)

    def forward_features(self, x):
        """Extract embedding vector (before classification head)."""
        return self.embedding(self.backbone(x))

    def forward(self, x):
        """Forward pass -> logits of shape (B, num_classes)."""
        features = self.forward_features(x)
        return self.fc(self.dropout(features))


class PRNUResNetLite(nn.Module):
    """Lightweight 4-layer CNN for quick experimentation."""

    def __init__(self, num_classes, in_channels=1, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward_features(self, x):
        return self.features(x).view(x.size(0), -1)

    def forward(self, x):
        return self.fc(self.dropout(self.forward_features(x)))


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, label_smoothing=0.0):
    """Train the model for one epoch. Returns dict with loss, accuracy."""
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, correct, total = 0.0, 0, 0

    for patches, labels in tqdm(loader, desc="Training", leave=False):
        patches, labels = patches.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += patches.size(0)

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1),
            "num_samples": total}


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model. Returns dict with loss, accuracy, top5, preds."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, top5_correct, total = 0.0, 0, 0, 0
    all_preds, all_labels = [], []

    for patches, labels in tqdm(loader, desc="Evaluating", leave=False):
        patches, labels = patches.to(device), labels.to(device)
        logits = model(patches)
        loss = criterion(logits, labels)
        total_loss += loss.item() * patches.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        if logits.size(1) >= 5:
            _, top5 = logits.topk(5, dim=1)
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            top5_correct += (preds == labels).sum().item()
        total += patches.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1),
            "top5_accuracy": top5_correct / max(total, 1), "num_samples": total,
            "all_preds": all_preds, "all_labels": all_labels}


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, path, optimizer=None, device=torch.device("cpu")):
    """Load model checkpoint. Returns the stored metrics dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("metrics", {})
