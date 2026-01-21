import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging
import os


def setup_logging(logger_name="Predictor"):
    """
    Configure logging for the module with both console and file handlers.
    
    Args:
        logger_name (str): Name of the logger instance.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get logger instance
    logger = logging.getLogger(logger_name)
    
    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(logs_dir, 'predict.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.debug(f"Logging initialized. Log file: {log_file_path}")
    
    return logger


# Initialize logger
logger = setup_logging("Predictor")

def predict_mask(model, image_path):
    """
    Predicts if a face has a mask or not using the trained model.

    Args:
        model (tf.keras.Model): Trained face mask detection model.
        image_path (str): Path to the image file.

    Returns:
        str: Prediction result - "With Mask" or "Without Mask".
        
    Raises:
        Exception: If image loading or prediction fails.
    """
    try:
        logger.info(f"Starting prediction for image: {image_path}")
        
        logger.debug("Loading image with size (224, 224)")
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        logger.debug("Image loaded successfully")
        
        logger.debug("Converting image to array")
        img_array = tf.keras.utils.img_to_array(img)
        logger.debug(f"Image array shape: {img_array.shape}")
        
        logger.debug("Adding batch dimension")
        img_array = tf.expand_dims(img_array, 0)
        logger.debug(f"Batch shape: {img_array.shape}")
        
        logger.debug("Applying MobileNetV2 preprocessing")
        img_array = preprocess_input(img_array)

        logger.debug("Running model prediction")
        predictions = model.predict(img_array, verbose=0)
        logger.debug(f"Raw prediction score: {predictions[0][0]:.4f}")
        
        logger.debug("Converting prediction to binary class (threshold=0.5)")
        prediction = (predictions > 0.5).astype(int).flatten()[0]
        result = "With Mask" if prediction == 1 else "Without Mask"
        
        logger.info(f"Prediction completed: {result} (confidence: {predictions[0][0]:.4f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise