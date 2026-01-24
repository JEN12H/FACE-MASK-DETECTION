import zipfile
import torch
from torchvision import datasets,transforms
import os 
from torch.utils.data import DataLoader
import logging 
import yaml



def setup_logging(logger_name = 'data_ingestion'):
    """
    Configure logging for the module with both console and file handlers.
    
    Args:
        logger_name (str): Name of the logger instance.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir,exist_ok = True)
    
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
    log_file_path = os.path.join(logs_dir,'data_ingestion.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.debug(f"Logging initialized. Log file: {log_file_path}")
    
    return logger


# Initialize logger
logger = setup_logging("data_ingestion")


import zipfile
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def extract_data_from_zip(zippath, extractpath):
    """
    Extract dataset from a ZIP file.
    
    Args:
        zippath (str): Path to the ZIP file containing the dataset
        extractpath (str): Path where the data should be extracted
    
    Returns:
        None
    """
    try:
        logger.info(f"Starting to extract ZIP file from: {zippath}")
        with zipfile.ZipFile(zippath, "r") as zip_ref:
            zip_ref.extractall(extractpath)
        logger.info(f"ZIP file extracted successfully to: {extractpath}")
        print("ZIP EXTRACTED SUCCESSFULLY")
    except FileNotFoundError as e:
        logger.error(f"ZIP file not found at: {zippath}")
        raise
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {str(e)}")
        raise


def get_transforms():
    """
    Define and return data transformations for training, validation, and testing.
    
    Returns:
        tuple: (train_transform, val_and_test_transforms)
    """
    logger.info("Creating data transformations...")
    try:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.debug("Training transformations created successfully")

        val_and_test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.debug("Validation and test transformations created successfully")
        logger.info("All transformations created successfully")
        print("SUCCESSFULLY COMPLETED TRANSFORMATION DEFINITION")
        return train_transform, val_and_test_transforms
    except Exception as e:
        logger.error(f"Error creating transformations: {str(e)}")
        raise


def load_datasets(dataset_root, train_transform, val_and_test_transforms):
    """
    Load training, validation, and test datasets.
    
    Args:
        dataset_root (str): Root path to the dataset
        train_transform: Transformations for training data
        val_and_test_transforms: Transformations for validation and test data
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Loading datasets from root: {dataset_root}")
    try:
        logger.info(f"Loading training dataset from: {dataset_root}/Train")
        train_dataset = ImageFolder(
            root=f"{dataset_root}/Train",
            transform=train_transform
        )
        logger.info(f"Training dataset loaded successfully. Size: {len(train_dataset)}")

        logger.info(f"Loading validation dataset from: {dataset_root}/Validation")
        val_dataset = ImageFolder(
            root=f"{dataset_root}/Validation",
            transform=val_and_test_transforms
        )
        logger.info(f"Validation dataset loaded successfully. Size: {len(val_dataset)}")

        logger.info(f"Loading test dataset from: {dataset_root}/Test")
        test_dataset = ImageFolder(
            root=f"{dataset_root}/Test",
            transform=val_and_test_transforms
        )
        logger.info(f"Test dataset loaded successfully. Size: {len(test_dataset)}")

        print("TRANSFORMATIONS APPLIED SUCCESSFULLY")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    except FileNotFoundError as e:
        logger.error(f"Dataset directory not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for training, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size for DataLoaders (default: 32)
        num_workers (int): Number of workers for data loading (default: 4)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders with batch_size={batch_size}, num_workers={num_workers}")
    try:
        logger.debug("Creating training data loader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Training data loader created. Batches: {len(train_loader)}")

        logger.debug("Creating validation data loader...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Validation data loader created. Batches: {len(val_loader)}")

        logger.debug("Creating test data loader...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Test data loader created. Batches: {len(test_loader)}")

        print("DATA-LOADERS CREATED SUCCESSFULLY")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise


def display_dataset_info(train_dataset):
    """
    Display information about the dataset classes.
    
    Args:
        train_dataset: Training dataset
    
    Returns:
        None
    """
    try:
        logger.info(f"Dataset classes: {train_dataset.classes}")
        logger.info(f"Class to index mapping: {train_dataset.class_to_idx}")
        print("Classes:", train_dataset.classes)
        print("Class to index:", train_dataset.class_to_idx)
    except Exception as e:
        logger.error(f"Error displaying dataset info: {str(e)}")
        raise


def load_data_pipeline(zippath, extractpath, dataset_root, batch_size=32, num_workers=4):
    """
    Complete data loading pipeline combining all steps.
    
    Args:
        zippath (str): Path to the ZIP file
        extractpath (str): Path to extract data
        dataset_root (str): Root path to the extracted dataset
        batch_size (int): Batch size for DataLoaders (default: 32)
        num_workers (int): Number of workers (default: 4)
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, 
                train_loader, val_loader, test_loader)
    """
    logger.info("Starting complete data loading pipeline...")
    try:
        # Extract data
        logger.info("Step 1: Extracting data from ZIP file")
        extract_data_from_zip(zippath, extractpath)
        
        # Get transformations
        logger.info("Step 2: Creating data transformations")
        train_transform, val_and_test_transforms = get_transforms()
        
        # Load datasets
        logger.info("Step 3: Loading datasets")
        train_dataset, val_dataset, test_dataset = load_datasets(
            dataset_root, 
            train_transform, 
            val_and_test_transforms
        )
        
        # Create data loaders
        logger.info("Step 4: Creating data loaders")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, 
            val_dataset, 
            test_dataset, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Display info
        logger.info("Step 5: Displaying dataset information")
        display_dataset_info(train_dataset)
        
        logger.info("Data loading pipeline completed successfully!")
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"Error in data loading pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    zippath = "/content/archive (5).zip"
    extractpath = "data"
    dataset_root = "/content/data/Face Mask Dataset"
    
    # Load complete pipeline
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_data_pipeline(
        zippath, 
        extractpath, 
        dataset_root
    )
    
    print("\nData ingestion completed successfully!")
