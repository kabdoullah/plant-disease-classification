"""Transformations, split stratifié et DataLoaders.

Point clé : un unique split d'indices est calculé puis sauvegardé, et
partagé par les deux modèles. Trois instances d'ImageFolder pointent vers
le même dossier mais appliquent des transformations différentes ;
seul le train reçoit l'augmentation.
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config


def build_transforms(train: bool):
    """Transformations d'entrée.

    Val/test : redimensionnement déterministe puis recadrage central.
    Train : augmentation modérée, préservant le signal pathologique
    (jitter de teinte volontairement très faible — la couleur des
    lésions porte l'information diagnostique).
    """
    normalize = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),   # 224 -> 256
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def make_split_indices(data_dir, seed=None, ratios=None, save_to=None):
    """Calcule (ou recharge) un split stratifié reproductible."""
    seed = seed if seed is not None else config.SEED
    ratios = ratios or config.SPLIT_RATIOS

    base = datasets.ImageFolder(data_dir)

    # Garde-fou critique : l'ordre d'ImageFolder DOIT correspondre à
    # config.CLASSES, sinon toutes les prédictions seront décalées.
    assert base.classes == config.CLASSES, (
        f"Ordre des classes incohérent.\n"
        f"ImageFolder : {base.classes}\nconfig     : {config.CLASSES}"
    )

    targets = [label for _, label in base.samples]
    indices = list(range(len(targets)))

    train_idx, rest_idx = train_test_split(
        indices, train_size=ratios[0], stratify=targets, random_state=seed
    )
    rest_targets = [targets[i] for i in rest_idx]
    val_share = ratios[1] / (ratios[1] + ratios[2])
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, stratify=rest_targets, random_state=seed
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(json.dumps(
            {"seed": seed, "ratios": list(ratios),
             "classes": base.classes, **splits}, indent=2))

    return base, splits


def get_dataloaders(data_dir, splits, batch_size=None, num_workers=None):
    """Construit les trois DataLoaders à partir d'un split existant."""
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    ds_train = datasets.ImageFolder(data_dir, transform=build_transforms(True))
    ds_eval = datasets.ImageFolder(data_dir, transform=build_transforms(False))

    loaders = {
        "train": DataLoader(Subset(ds_train, splits["train"]),
                            batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=False),
        "val": DataLoader(Subset(ds_eval, splits["val"]),
                          batch_size=config.BATCH_SIZE_EVAL, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(Subset(ds_eval, splits["test"]),
                           batch_size=config.BATCH_SIZE_EVAL, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }
    return loaders