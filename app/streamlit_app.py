"""Interface de démonstration — consomme l'API FastAPI."""
import base64
import io

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8001"

st.set_page_config(page_title="Diagnostic foliaire", page_icon="🍃", layout="wide")
st.title("🍃 Détection de maladies des feuilles de tomate")
st.caption("Classification par Vision Transformer — projet M1 IA, DIT")

# --- État de l'API
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("status") == "ok":
        st.sidebar.success(f"API connectée — {health['model']}")
    else:
        st.sidebar.error("Modèle non chargé côté API")
except requests.RequestException:
    st.sidebar.error("API injoignable")
    st.warning("Démarrez l'API : `uvicorn api.main:app --port 8000`")
    st.stop()

explain = st.sidebar.checkbox("Afficher l'interprétabilité", value=True)

uploaded = st.file_uploader("Image de feuille", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image analysée")
        st.image(Image.open(uploaded), use_container_width=True)

    if st.button("Analyser", type="primary", use_container_width=True):
        uploaded.seek(0)
        with st.spinner("Analyse en cours…"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded, uploaded.type)},
                    params={"explain": explain}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"Échec de la requête : {exc}")
                st.stop()

        with col2:
            st.subheader("Résultat")
            conf = result["confidence"]
            st.metric("Maladie prédite", result["predicted_label_fr"],
                      f"confiance {conf:.1%}")

            if conf < 0.60:
                st.warning("Confiance faible — résultat à interpréter avec prudence.")

            st.caption(f"Modèle : {result['model']} · "
                       f"inférence {result['inference_ms']} ms")

            st.markdown("**Distribution des probabilités**")
            for p in result["probabilities"][:3]:
                st.progress(p["probability"],
                            text=f"{p['label_fr']} — {p['probability']:.1%}")

        if explain and "explanation" in result:
            expl = result["explanation"]
            st.divider()
            st.subheader("Interprétabilité")
            if "image_base64" in expl:
                st.image(Image.open(io.BytesIO(base64.b64decode(expl["image_base64"]))),
                         caption=f"{expl['method']} — zones les plus influentes",
                         width=400)
                st.caption("Les régions chaudes indiquent les zones sur lesquelles "
                           "le modèle a concentré son attention.")
            else:
                st.info(expl.get("error", "Explication indisponible"))

st.divider()
st.caption("⚠️ Outil de démonstration académique. Le modèle est entraîné sur des "
           "images de laboratoire (PlantVillage) et n'est pas validé pour un "
           "usage agricole réel.")