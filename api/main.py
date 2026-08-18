"""API d'inférence — modèle retenu : ViT-Tiny (cf. décision OS9)."""
import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.explainability.xai import attention_rollout_vit, denormalize, overlay
from src.models.build import build_model
from src.utils import load_checkpoint

STATE = {}
MODEL_KIND = "vit"
MAX_UPLOAD_MB = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois, au démarrage du serveur."""
    ckpt_path = config.MODELS_DIR / f"{MODEL_KIND}_best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Poids introuvables : {ckpt_path}")

    device = torch.device("cpu")            # déploiement CPU assumé
    model, _ = build_model(MODEL_KIND, pretrained=False)
    ckpt = load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    meta = ckpt["metadata"]
    STATE.update(
        model=model,
        device=device,
        classes=meta["classes"],            # ordre lu depuis le checkpoint
        labels_fr=config.CLASS_LABELS_FR,
        model_name="ViT-Tiny",
        backbone=meta.get("backbone", config.VIT_BACKBONE),
        val_acc=meta.get("val_acc"),
        transform=transforms.Compose([
            transforms.Resize(int(meta["img_size"] * 1.14)),
            transforms.CenterCrop(meta["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ]),
    )
    print(f"✅ Modèle chargé : {STATE['model_name']} ({len(STATE['classes'])} classes)")
    yield
    STATE.clear()


app = FastAPI(
    title="API de classification de maladies foliaires",
    description="Détection de maladies de la tomate par Vision Transformer",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    """Vérifie que le modèle est chargé et prêt."""
    return {
        "status": "ok" if "model" in STATE else "model_not_loaded",
        "model": STATE.get("model_name"),
        "num_classes": len(STATE.get("classes", [])),
    }


@app.get("/classes")
def classes():
    return {"classes": [
        {"id": i, "name": c, "label_fr": STATE["labels_fr"].get(c, c)}
        for i, c in enumerate(STATE["classes"])
    ]}


def _read_image(raw: bytes) -> Image.Image:
    """Décode et valide une image envoyée par le client."""
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image trop volumineuse (max {MAX_UPLOAD_MB} Mo)")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Fichier illisible ou format d'image non supporté")


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = False):
    """Prédit la maladie d'une feuille.

    explain=True ajoute une carte d'attention encodée en base64.
    """
    if "model" not in STATE:
        raise HTTPException(503, "Modèle non chargé")

    image = _read_image(await file.read())
    tensor = STATE["transform"](image)

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(STATE["model"](tensor.unsqueeze(0)), dim=1)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    name = STATE["classes"][idx]

    response = {
        "predicted_class": name,
        "predicted_label_fr": STATE["labels_fr"].get(name, name),
        "confidence": round(float(probs[idx]), 4),
        "model": STATE["model_name"],
        "inference_ms": round(inference_ms, 1),
        "probabilities": sorted(
            [{"class": c, "label_fr": STATE["labels_fr"].get(c, c),
              "probability": round(float(p), 4)}
             for c, p in zip(STATE["classes"], probs)],
            key=lambda d: d["probability"], reverse=True),
    }

    if explain:
        try:
            heatmap, _ = attention_rollout_vit(STATE["model"], tensor)
            blended = overlay(denormalize(tensor), heatmap)
            buf = io.BytesIO()
            Image.fromarray((blended * 255).astype(np.uint8)).save(buf, format="PNG")
            response["explanation"] = {
                "method": "Attention rollout",
                "image_base64": base64.b64encode(buf.getvalue()).decode(),
            }
        except Exception as exc:
            response["explanation"] = {"error": f"Explication indisponible : {exc}"}

    return response