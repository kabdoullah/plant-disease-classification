"""Boucle d'entraînement mutualisée par les deux architectures.

Le même code entraîne ResNet18 et ViT-Tiny : c'est cette mutualisation
qui garantit que la comparaison porte sur les architectures et non sur
des différences de protocole.
"""
import time

import torch
import torch.nn as nn

from src import config
from src.metadata import build_metadata
from src.utils import save_checkpoint


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Perte et accuracy sur un jeu donné, sans calcul de gradients."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Une epoch d'entraînement.

    ATTENTION : le modèle reste en mode eval() (cf. rapport § II.3).
    Le backbone est gelé ; le laisser en train() ferait dériver les
    statistiques BatchNorm de ResNet18 alors que le LayerNorm du ViT
    resterait figé, rompant la symétrie du protocole.
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def fit(model, trainable_params, loaders, device, kind, models_dir,
        epochs=None, lr=None):
    """Entraîne, valide, sauvegarde les checkpoints après CHAQUE epoch.

    Deux fichiers sont maintenus :
      - <kind>_last.pt : état complet, permet de reprendre après
        une déconnexion Colab ;
      - <kind>_best.pt : meilleure accuracy de validation observée,
        c'est ce fichier qui sera évalué puis déployé.
    """
    epochs = epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    started = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc = evaluate(
            model, loaders["val"], criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        flag = ""

        meta = build_metadata(
            kind,
            backbone=config.CNN_BACKBONE if kind == "cnn" else config.VIT_BACKBONE,
            epochs_planned=epochs,
            lr=lr,
            batch_size=config.BATCH_SIZE,
        )

        # Checkpoint de reprise : écrit systématiquement
        save_checkpoint(models_dir / f"{kind}_last.pt",
                        model, optimizer, epoch, history, meta)

        # Meilleur modèle : uniquement si la validation progresse
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            meta["val_acc"] = va_acc
            meta["best_epoch"] = epoch
            save_checkpoint(models_dir / f"{kind}_best.pt",
                            model, None, epoch, history, meta)
            flag = "  ← meilleur"

        print(f"Epoch {epoch}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{elapsed:.1f}s{flag}")

    total = time.time() - started
    print(f"\nTerminé en {total:.1f}s — meilleure val acc : {best_val_acc:.4f}")
    if history["val_acc"].index(max(history["val_acc"])) == epochs - 1:
        print("⚠️  Meilleure val acc atteinte à la DERNIÈRE epoch "
              "— le modèle n'a probablement pas convergé.")
    return history, best_val_acc, total