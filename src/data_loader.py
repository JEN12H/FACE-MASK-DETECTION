import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import zipfile
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def extract_zip_if_needed(zip_path, extract_to):
    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        if not os.path.exists(extract_to) or not os.listdir(extract_to):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
        else:
            print(f"Data already extracted in {extract_to}")

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
    """
    if zip_path:
        extract_zip_if_needed(zip_path, os.path.dirname(train_dir))

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
    )

    return train_data, test_data, validation_data

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
    Applies MobileNetV2 preprocessing.

    Args:
        dataset (tf.data.Dataset)
        augment (bool): Apply augmentation (True for training only)

    Returns:
        tf.data.Dataset
    """
    if augment:
        augmenter = data_augmentation()
        dataset = dataset.map(lambda x, y: (preprocess_input(augmenter(x)), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def optimize_performance(dataset):
    """
    Improves performance using caching and prefetching.
    """
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)