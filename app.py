
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model("guava_cnn_model.keras")

# Guava classes
class_names = ["Allahabadi", "Gola", "Surahi"]

st.set_page_config(
    page_title="Guava Variety Classifier",
    page_icon="🍈"
)

st.title("🍈 Guava Variety Classifier")
st.write("Upload a guava image to identify its variety.")

uploaded_file = st.file_uploader(
    "Choose a guava image",
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

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.success(f"Predicted Variety: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(class_names):
        st.write(
            f"{class_name}: {predictions[0][i] * 100:.2f}%"
        )
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(
    page_title="Guava Variety Classifier",
    page_icon="🍈",
    layout="centered"
)

st.title("🍈 Guava Variety Classifier")
st.write("Upload a guava image to identify its variety.")

# Hugging Face model
MODEL_URL = "https://huggingface.co/Gulafroz/guava-cnn-model/resolve/main/guava_cnn_model.keras"
MODEL_PATH = "guava_cnn_model.keras"

@st.cache_resource
def load_model():
    # Download model from Hugging Face if not already present
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Classes used during training
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

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    if st.button("🔍 Predict Variety"):

        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100

        st.success(
            f"### Predicted Variety: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for i, name in enumerate(class_names):
            probability = float(predictions[0][i]) * 100
            st.write(f"{name}: {probability:.2f}%")
            st.progress(min(int(probability), 100))
