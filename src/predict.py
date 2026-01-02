import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_mask(model, image_path):
    """
    Predicts if a face has a mask or not.

    Args:
        model (tf.keras.Model): Trained model.
        image_path (str): Path to the image.

    Returns:
        str: Prediction result.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    prediction = (predictions > 0.5).astype(int).flatten()[0]

    return "With Mask" if prediction == 1 else "Without Mask"