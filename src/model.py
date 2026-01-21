import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import logging
import os


def setup_logging(logger_name="ModelBuilder"):
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
    log_file_path = os.path.join(logs_dir, 'model.log')
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
logger = setup_logging("ModelBuilder")

def build_model(input_shape=(224, 224, 3)):
    """
    Builds the MobileNetV2-based model for face mask detection.

    Args:
        input_shape (tuple): Input shape for the model.

    Returns:
        tuple: (compiled_model, base_model)
        
    Raises:
        Exception: If model building or compilation fails.
    """
    try:
        logger.info(f"Starting model building with input shape {input_shape}")
        
        logger.debug("Loading MobileNetV2 base model from ImageNet weights")
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        logger.info("Base model loaded and frozen successfully")

        logger.debug("Building custom layers on top of base model")
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        logger.debug("Added GlobalAveragePooling2D layer")
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        logger.debug("Added Dense layer with 128 units")
        
        x = tf.keras.layers.Dropout(0.5)(x)
        logger.debug("Added Dropout layer with rate 0.5")
        
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        logger.debug("Added output Dense layer with sigmoid activation")

        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        logger.info("Model architecture created successfully")

        logger.debug("Compiling model with Adam optimizer (lr=1e-3)")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model compiled successfully")
        
        return model, base_model
        
    except Exception as e:
        logger.error(f"Error building model: {str(e)}", exc_info=True)
        raise

def fine_tune_model(model, base_model):
    """
    Fine-tunes the model by unfreezing some layers of the base model.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        base_model (tf.keras.Model): The base model.

    Returns:
        tf.keras.Model: Fine-tuned and recompiled model.
        
    Raises:
        Exception: If fine-tuning or recompilation fails.
    """
    try:
        logger.info("Starting fine-tuning process")
        
        logger.debug("Unfreezing base model layers")
        base_model.trainable = True

        logger.debug("Freezing first 30 layers of base model")
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        logger.info(f"Froze first 30 layers, unfroze remaining {len(base_model.layers) - 30} layers")

        logger.debug("Recompiling model with lower learning rate (lr=1e-5) for fine-tuning")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model recompiled successfully for fine-tuning")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        raise