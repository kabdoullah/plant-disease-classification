"""Utilitaires transverses : graines, device, checkpoints, mesure de taille."""
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa pour rendre les runs comparables.

    Couvre : Python (random), NumPy, PyTorch CPU et CUDA, ainsi que le hash
    seed utilisé par certaines structures Python. Indispensable pour que la
    comparaison CNN/ViT porte sur les architectures et non sur le bruit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Retourne le GPU s'il existe, sinon le CPU.

    Permet au même code de tourner sur Colab (cuda) et en local (cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, epoch, history, metadata) -> None:
    """Sauvegarde ATOMIQUE d'un checkpoint.

    L'écriture se fait dans un fichier temporaire, puis on le renomme.
    Motivation : si la session Colab meurt pendant l'écriture d'un gros
    fichier, un `torch.save` direct laisse un .pt tronqué et illisible —
    on perd alors le checkpoint précédent ET le nouveau. Le renommage
    étant une opération quasi instantanée, la fenêtre de corruption
    devient négligeable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "history": history,
            "metadata": metadata,   # contrat auto-descriptif (cf. §2.6)
        },
        tmp,
    )
    shutil.move(str(tmp), str(path))


def load_checkpoint(path, model=None, optimizer=None, device=None):
    """Charge un checkpoint, en local (CPU) comme sur Colab (GPU).

    `map_location` est OBLIGATOIRE : sans lui, un checkpoint écrit depuis
    un GPU tente de se recharger sur cuda:0 et lève une erreur sur une
    machine sans GPU — exactement votre cas en local.
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def count_parameters(model, trainable_only: bool = False) -> int:
    """Compte les paramètres — servira aux métriques de complexité (OS6)."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def file_size_mb(path) -> float:
    """Taille d'un fichier en Mo — métrique d'empreinte (OS6, NF2)."""
    return Path(path).stat().st_size / (1024 ** 2)