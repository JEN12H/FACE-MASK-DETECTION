import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import zipfile
import os
import logging
import yaml


def setup_logging(logger_name="DataLoader"):
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
    log_file_path = os.path.join(logs_dir, 'data_loader.log')
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
logger = setup_logging("DataLoader")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def extract_zip_if_needed(zip_path, extract_to):
    """
    Extracts zip file if it exists and target directory is empty.
    
    Args:
        zip_path (str): Path to zip file.
        extract_to (str): Directory to extract to.
    """
    try:
        if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
            if not os.path.exists(extract_to) or not os.listdir(extract_to):
                logger.info(f"Extracting {zip_path} to {extract_to}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info(f"Successfully extracted data")
            else:
                logger.info(f"Data already exists in {extract_to}, skipping extraction")
        else:
            logger.warning(f"Zip file not found or invalid: {zip_path}")
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        raise

def load_datasets(train_dir, test_dir, val_dir, zip_path=None):
    """
    Loads training, test, and validation datasets.

    Args:
        train_dir (str): Path to training data directory.
        test_dir (str): Path to test data directory.
        val_dir (str): Path to validation data directory.
        zip_path (str): Path to zip file to extract if needed.

    Returns:
        tuple: (train_data, test_data, validation_data)
        
    Raises:
        Exception: If datasets cannot be loaded from directories.
    """
    try:
        logger.info("Starting dataset loading process")
        
        if zip_path:
            logger.info(f"Extracting zip file if needed: {zip_path}")
            extract_zip_if_needed(zip_path, os.path.dirname(train_dir))

        logger.info(f"Loading training data from {train_dir}")
        train_data = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
            shuffle=True
        )
        logger.info(f"Training data loaded successfully")

        logger.info(f"Loading test data from {test_dir}")
        test_data = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
        )
        logger.info(f"Test data loaded successfully")

        logger.info(f"Loading validation data from {val_dir}")
        validation_data = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
        )
        logger.info(f"Validation data loaded successfully")
        logger.info("All datasets loaded successfully")

        return train_data, test_data, validation_data
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}", exc_info=True)
        raise

def data_augmentation():
    """
    Returns data augmentation pipeline for training data.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.1)
    ])

def preprocess_dataset(dataset, augment=False):
    """
    Applies MobileNetV2 preprocessing and optional augmentation.

    Args:
        dataset (tf.data.Dataset): Input dataset.
        augment (bool): Apply augmentation (True for training only).

    Returns:
        tf.data.Dataset: Preprocessed dataset.
    """
    try:
        logger.debug(f"Preprocessing dataset with augment={augment}")
        if augment:
            logger.debug("Applying data augmentation pipeline")
            augmenter = data_augmentation()
            dataset = dataset.map(lambda x, y: (preprocess_input(augmenter(x)), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        logger.debug("Dataset preprocessing completed")
        return dataset
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}", exc_info=True)
        raise

def optimize_performance(dataset):
    """
    Improves performance using caching and prefetching.
    
    Args:
        dataset (tf.data.Dataset): Input dataset.
        
    Returns:
        tf.data.Dataset: Optimized dataset.
    """
    try:
        logger.debug("Optimizing dataset performance with caching and prefetching")
        optimized = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        logger.debug("Dataset optimization completed")
        return optimized
    except Exception as e:
        logger.error(f"Error optimizing dataset: {str(e)}", exc_info=True)
        raise