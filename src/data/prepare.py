"""Constitution du sous-ensemble tomate à partir de PlantVillage brut.

Responsabilité unique : à partir d'un dossier PlantVillage quelconque,
produire une arborescence propre et normalisée :

    out_root/
        Tomato___Bacterial_spot/   (400 images)
        Tomato___Early_blight/     (400 images)
        ...
"""
import json
import random
import shutil
from pathlib import Path

from src import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _normalize(name: str) -> str:
    """Réduit un nom de dossier à sa forme canonique comparable.

    'Tomato___Bacterial_spot', 'Tomato_Bacterial_spot' et
    'tomato bacterial spot' donnent tous 'tomatobacterialspot'.
    Rend le code robuste aux variations de nommage entre les
    différentes distributions de PlantVillage.
    """
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_class_dirs(raw_root) -> dict:
    """Associe chaque classe canonique au dossier réel correspondant."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_root}")

    index = {_normalize(d.name): d for d in raw_root.iterdir() if d.is_dir()}

    resolved, missing = {}, []
    for canonical in config.CLASSES:
        match = index.get(_normalize(canonical))
        if match is None:
            missing.append(canonical)
        else:
            resolved[canonical] = match

    if missing:
        raise FileNotFoundError(
            f"Classes non trouvées : {missing}\n"
            f"Dossiers disponibles : {sorted(d.name for d in raw_root.iterdir() if d.is_dir())}"
        )
    return resolved


def inventory(raw_root) -> dict:
    """Compte les images disponibles par classe AVANT sous-échantillonnage.

    Ces chiffres illustrent le déséquilibre naturel du jeu de données
    et doivent figurer dans l'analyse exploratoire du rapport.
    """
    return {
        canonical: len([p for p in src.iterdir() if p.suffix in IMG_EXT])
        for canonical, src in resolve_class_dirs(raw_root).items()
    }


def build_subset(raw_root, out_root, max_per_class=None, seed=None) -> dict:
    """Copie un échantillon aléatoire de chaque classe vers out_root.

    Le tirage est reproductible (seed) et les dossiers de sortie portent
    les noms canoniques de config.CLASSES, ce qui garantit que
    ImageFolder produira exactement l'ordre de classes attendu.
    """
    max_per_class = max_per_class or config.MAX_PER_CLASS
    seed = seed if seed is not None else config.SEED

    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    rng = random.Random(seed)
    stats = {}

    for canonical, src_dir in resolve_class_dirs(raw_root).items():
        images = sorted(p for p in src_dir.iterdir() if p.suffix in IMG_EXT)
        stats[canonical] = {"available": len(images)}

        rng.shuffle(images)
        selected = images[:max_per_class]

        dst_dir = out_root / canonical      # nom canonique imposé
        dst_dir.mkdir()
        for img in selected:
            shutil.copy2(img, dst_dir / img.name)

        stats[canonical]["selected"] = len(selected)

    (out_root / "_subset_info.json").write_text(
        json.dumps({"seed": seed, "max_per_class": max_per_class, "stats": stats}, indent=2)
    )
    return stats