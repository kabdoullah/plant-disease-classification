"""Évaluation sur le jeu de test et comparaison des deux architectures."""
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src import config


@torch.no_grad()
def predict_all(model, loader, device):
    """Retourne (y_true, y_pred, y_proba) sur l'ensemble d'un loader."""
    model.eval().to(device)
    trues, preds, probas = [], [], []

    for images, labels in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1)
        probas.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
        trues.append(labels.numpy())

    return (np.concatenate(trues), np.concatenate(preds),
            np.concatenate(probas))


def compute_metrics(y_true, y_pred) -> dict:
    """Métriques globales en moyenne macro (classes traitées à égalité)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_report(y_true, y_pred):
    """Rapport détaillé par classe, avec libellés français."""
    labels = [config.CLASS_LABELS_FR[c] for c in config.CLASSES]
    return classification_report(y_true, y_pred, target_names=labels,
                                 digits=4, zero_division=0)


def get_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))


def mcnemar_table(y_true, pred_a, pred_b) -> dict:
    """Table de contingence appariée entre deux modèles.

    b : A seul a raison ; c : B seul a raison.
    Ces deux cases sont les seules informatives (cf. test de McNemar).
    """
    ok_a, ok_b = (pred_a == y_true), (pred_b == y_true)
    return {
        "both_correct":  int((ok_a & ok_b).sum()),
        "only_a":        int((ok_a & ~ok_b).sum()),   # b
        "only_b":        int((~ok_a & ok_b).sum()),   # c
        "both_wrong":    int((~ok_a & ~ok_b).sum()),
    }


def benchmark_cpu(model, loader, n_samples=100, n_warmup=10) -> dict:
    """Temps d'inférence image par image sur CPU.

    Régime batch_size=1, celui d'une API traitant une requête à la fois.
    Passes de chauffe exclues ; médiane rapportée (robuste aux à-coups).
    """
    device = torch.device("cpu")
    model.eval().to(device)

    images = []
    for batch, _ in loader:
        for img in batch:
            images.append(img.unsqueeze(0))
            if len(images) >= n_samples + n_warmup:
                break
        if len(images) >= n_samples + n_warmup:
            break

    with torch.no_grad():
        for img in images[:n_warmup]:
            model(img)

        timings = []
        for img in images[n_warmup:]:
            t0 = time.perf_counter()
            model(img)
            timings.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(timings)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "n":         len(arr),
    }