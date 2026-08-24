import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(
    page_title="Guava Variety Classifier",
    page_icon="🍈"
)

st.title("🍈 Guava Variety Classifier")
st.write("Upload a guava image to identify its variety.")

MODEL_URL = "https://huggingface.co/Gulafroz/guava-cnn-model/resolve/main/guava_cnn_model.keras"
MODEL_PATH = "/tmp/guava_cnn_model.keras"

@st.cache_resource
def load_guava_model():

    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_guava_model()

class_names = ["Allahabadi", "Gola", "Surahi"]

uploaded_file = st.file_uploader(
    "Upload Guava Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Guava Image",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    if st.button("🔍 Predict Variety"):

        predictions = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        st.success(
            f"Predicted Variety: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for i, name in enumerate(class_names):
            probability = predictions[0][i] * 100
            st.write(f"{name}: {probability:.2f}%")
