import logging
import os


def setup_logging(logger_name="Trainer"):
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
    log_file_path = os.path.join(logs_dir, 'train.log')
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
logger = setup_logging("Trainer")


def train_model(model, train_data, validation_data, epochs=5):
    """
    Trains the model for the specified number of epochs.

    Args:
        model (tf.keras.Model): The model to train.
        train_data (tf.data.Dataset): Training dataset.
        validation_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs. Default is 5.

    Returns:
        tf.keras.callbacks.History: Training history object containing metrics.
        
    Raises:
        Exception: If training fails.
    """
    try:
        logger.info(f"Starting model training for {epochs} epochs")
        logger.debug(f"Training data batches available")
        logger.debug(f"Validation data batches available")
        
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs
        )
        
        logger.info(f"Training completed successfully after {epochs} epochs")
        logger.debug(f"Final training metrics: {history.history}")
        
        return history
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

def fine_tune_model(model, train_data, validation_data, epochs=5):
    """
    Fine-tunes the model with unfrozen layers for additional epochs.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        train_data (tf.data.Dataset): Training dataset.
        validation_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of fine-tuning epochs. Default is 5.

    Returns:
        tf.keras.callbacks.History: Fine-tuning history object containing metrics.
        
    Raises:
        Exception: If fine-tuning fails.
    """
    try:
        logger.info(f"Starting model fine-tuning for {epochs} epochs")
        logger.debug(f"Training with lower learning rate on unfrozen base model layers")
        
        fine_tune_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs
        )
        
        logger.info(f"Fine-tuning completed successfully after {epochs} epochs")
        logger.debug(f"Fine-tuning metrics: {fine_tune_history.history}")
        
        return fine_tune_history
        
    except Exception as e:
        logger.error(f"Error during model fine-tuning: {str(e)}", exc_info=True)
        raise