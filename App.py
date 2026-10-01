"""Streamlit app: upload a brain MRI and get the predicted tumour type + confidence.

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma",
          "no_tumor": "No tumour", "pituitary": "Pituitary tumour"}
MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model(name: str):
    from tensorflow import keras           # imported lazily so the UI loads fast
    return keras.models.load_model(MODEL_DIR / f"{name}.h5", compile=False)


def predict(model, img: Image.Image) -> np.ndarray:
    """Resize to 224x224 and predict. Rescaling/preprocessing is inside the saved model."""
    arr = np.asarray(img.convert("RGB").resize((224, 224)), dtype="float32")[None]
    return model.predict(arr, verbose=0)[0]


st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Classifies an MRI slice as glioma, meningioma, pituitary tumour or no tumour.")

available = sorted(p.stem for p in MODEL_DIR.glob("*.h5"))
if not available:
    st.error("No trained models found in `models/`. Run `python src/train.py` first.")
    st.stop()

default = available.index("efficientnetb0") if "efficientnetb0" in available else 0
choice = st.sidebar.selectbox("Model", available, index=default)
st.sidebar.info("Research/education demo only - not a medical device. "
                "Predictions must not replace review by a qualified radiologist.")

file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
if file:
    try:
        img = Image.open(file)
        st.image(img, caption="Uploaded MRI", use_container_width=True)
        with st.spinner("Analysing..."):
            probs = predict(load_model(choice), img)
        top = int(probs.argmax())
        st.subheader(f"Prediction: {LABELS[CLASS_NAMES[top]]}")
        st.metric("Confidence", f"{probs[top]:.1%}")
        if probs[top] < 0.70:
            st.warning("Low confidence - treat this result with extra caution.")
        st.bar_chart({LABELS[c]: float(p) for c, p in zip(CLASS_NAMES, probs)})
    except Exception as e:                  # corrupt file, wrong format, model error
        st.error(f"Could not process this image: {e}")
