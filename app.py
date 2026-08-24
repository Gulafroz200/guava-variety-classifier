
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
