# app.py
import io
import numpy as np
from PIL import Image

import streamlit as st
from tensorflow import keras


st.set_page_config(
    page_title="Diagnostic maladies des plantes",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Diagnostic de maladies des plantes")
st.write(
    "Charge une photo de feuille de **Peach / Pepper bell / Potato** "
    "et le modèle prédira la plante et la maladie."
)


@st.cache_resource
def load_model():
    model = keras.models.load_model("models/deep_cnn.h5")
    return model

model = load_model()

CLASS_NAMES = {
    0: "Peach - Bacterial spot",
    1: "Peach - Healthy",
    2: "Pepper bell - Bacterial spot",
    3: "Pepper bell - Healthy",
    4: "Potato - Early blight",
    5: "Potato - Healthy",
    6: "Potato - Late blight",
}

IMG_SIZE = 128  


def preprocess_image(image: Image.Image) -> np.ndarray:

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr



uploaded_file = st.file_uploader(
    "Drop or choose a leaf image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    help="Use clear photos, centered on the leaf."
)

if uploaded_file is not None:
    # preview of the image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing the image..."):
                # simple progress bar
                progress = st.progress(0)
                x = preprocess_image(image)
                progress.progress(50)

                # prediction
                preds = model.predict(x)
                progress.progress(100)

            # results
            probas = preds[0]
            top_idx = int(np.argmax(probas))
            top_label = CLASS_NAMES.get(top_idx, f"Class {top_idx}")
            top_conf = float(probas[top_idx])

            st.success(f"RResult: **{top_label}**")
            st.write(f"Confidence: **{top_conf*100:.1f} %**")
            st.markdown("### Détail des probabilités")
            for idx, p in enumerate(probas):
                st.write(f"- {CLASS_NAMES.get(idx, f'Classe {idx}')} : {p*100:.1f} %")

else:
    st.info("Charge une image pour commencer le diagnostic.")

