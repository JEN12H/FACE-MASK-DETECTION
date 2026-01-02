import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.predict import predict_mask

# Set page config
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")

# Load the model with caching
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenetv2_mask_detector.h5')
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("😷 Face Mask Detection")
st.write("Upload an image to detect if a person is wearing a face mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    
    with st.spinner("Classifying..."):
        # Save temporarily using tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            image.save(temp_path)
        
        try:
            prediction = predict_mask(model, temp_path)
            
            # Display result with color
            if "With Mask" in prediction:
                st.success(f"✅ Prediction: **{prediction}**")
            else:
                st.warning(f"⚠️ Prediction: **{prediction}**")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)