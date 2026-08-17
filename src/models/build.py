"""Construction des deux architectures comparées.

Les deux fonctions suivent le même contrat : charger un backbone
pré-entraîné, remplacer la tête de classification par une couche adaptée
au nombre de classes du projet, et geler l'intégralité du backbone.
"""
import timm
import torch.nn as nn
from torchvision import models

from src import config


def build_cnn(num_classes=None, pretrained=True):
    """ResNet18 pré-entraîné sur ImageNet-1k, backbone gelé."""
    num_classes = num_classes or config.NUM_CLASSES
    weights = config.CNN_WEIGHTS if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)   # tête neuve
    return model


def build_vit(num_classes=None, pretrained=True):
    """ViT-Tiny patch16 224, backbone gelé.

    timm remplace automatiquement la tête lorsqu'on précise num_classes.
    """
    num_classes = num_classes or config.NUM_CLASSES
    return timm.create_model(
        config.VIT_BACKBONE,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_head(model, kind: str) -> nn.Module:
    """Retourne le module de classification (le seul entraînable)."""
    if kind == "cnn":
        return model.fc
    if kind == "vit":
        return model.get_classifier()
    raise ValueError(f"kind inconnu : {kind}")


def freeze_backbone(model, kind: str):
    """Gèle tout sauf la tête de classification.

    Retourne la liste des paramètres à confier à l'optimiseur.
    """
    for p in model.parameters():
        p.requires_grad = False
    head = get_head(model, kind)
    for p in head.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def build_model(kind: str, num_classes=None, pretrained=True):
    """Fabrique unifiée : construit, gèle, retourne (modèle, params entraînables)."""
    model = build_cnn(num_classes, pretrained) if kind == "cnn" \
        else build_vit(num_classes, pretrained)
    trainable = freeze_backbone(model, kind)
    return model, trainable