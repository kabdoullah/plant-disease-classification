"""Construction du bloc de métadonnées embarqué dans chaque checkpoint."""
from datetime import datetime

from src import config


def build_metadata(model_name: str, **extra) -> dict:
    """Rend le fichier de poids auto-suffisant.

    Toute personne (ou toute API) chargeant le .pt dispose alors de l'ordre
    exact des classes, de la taille d'entrée et des constantes de
    normalisation, sans avoir à consulter le code d'entraînement.
    """
    meta = {
        "model_name": model_name,
        "classes": config.CLASSES,          # ORDRE CRITIQUE
        "num_classes": config.NUM_CLASSES,
        "img_size": config.IMG_SIZE,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
        "seed": config.SEED,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(extra)
    return meta