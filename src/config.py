"""Configuration centrale du projet.

Point de vérité unique : toute constante utilisée à plus d'un endroit
(classes, taille d'image, normalisation, chemins) est définie ici.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Reproductibilité
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Classes — l'ORDRE DE CETTE LISTE FAIT FOI POUR TOUT LE PROJET.
# Il correspond à l'ordre alphabétique produit par torchvision.ImageFolder,
# ce qui garantit la cohérence entre entraînement et inférence.
# Ne jamais réordonner après le premier entraînement.
# --------------------------------------------------------------------------
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

CLASS_LABELS_FR = {
    "Tomato___Bacterial_spot":     "Tache bactérienne",
    "Tomato___Early_blight":       "Alternariose (mildiou précoce)",
    "Tomato___Late_blight":        "Mildiou tardif",
    "Tomato___Septoria_leaf_spot": "Septoriose",
    "Tomato___healthy":            "Feuille saine",
}

NUM_CLASSES = len(CLASSES)

# --------------------------------------------------------------------------
# Prétraitement — imposé par les backbones pré-entraînés sur ImageNet.
# ResNet18 et ViT-Tiny patch16 attendent tous deux du 224x224 normalisé
# avec les statistiques d'ImageNet. Utiliser d'autres valeurs dégraderait
# silencieusement les performances.
# --------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Répartition des données
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
NUM_WORKERS  = 2                    # 2 sur Colab, 0 en local

# --------------------------------------------------------------------------
# Backbones — noms figés après vérification de disponibilité (timm 1.0.28).
# Le ViT retenu est affiné sur ImageNet-1k pour aligner le domaine de
# pré-entraînement avec celui de ResNet18 (IMAGENET1K_V1).
# --------------------------------------------------------------------------
CNN_BACKBONE = "resnet18"
CNN_WEIGHTS  = "IMAGENET1K_V1"
VIT_BACKBONE = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
# --------------------------------------------------------------------------
# Hyperparamètres (valeurs de départ, ajustables aux étapes 5 et 6)
# --------------------------------------------------------------------------
BATCH_SIZE      = 32   # entraînement sur GPU T4
BATCH_SIZE_EVAL = 64
NUM_EPOCHS      = 15
LEARNING_RATE   = 1e-3  # élevé car seule la tête est entraînée
MAX_PER_CLASS   = 400   # sous-échantillonnage (étape 3)

# --------------------------------------------------------------------------
# Chemins — résolus automatiquement selon l'environnement
# --------------------------------------------------------------------------
IN_COLAB = Path("/content").exists()

if IN_COLAB:
    DRIVE_ROOT   = Path("/content/drive/MyDrive/plant-disease")
    DATA_DIR     = Path("/content/data")          # disque VM : rapide
    MODELS_DIR   = DRIVE_ROOT / "models"          # Drive : persistant
    RESULTS_DIR  = DRIVE_ROOT / "results"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR     = PROJECT_ROOT / "data"
    MODELS_DIR   = PROJECT_ROOT / "models"
    RESULTS_DIR  = PROJECT_ROOT / "results"