import torch
import torch.nn as nn
from torchvision import models
import logging

# Setup logging
logger = logging.getLogger("model")


def load_pretrained_model(num_classes=2, freeze_base=True, device='cpu'):
    """
    Load a pre-trained MobileNetV2 model and prepare it for transfer learning.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        freeze_base (bool): Whether to freeze base model layers (default: True)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Modified MobileNetV2 model ready for training
    """
    logger.info("Loading pre-trained MobileNetV2 model...")
    try:
        # Load pre-trained MobileNetV2
        model = models.mobilenet_v2(pretrained=True)
        logger.info("MobileNetV2 model loaded successfully with ImageNet weights")
        
        # Freeze base model layers for transfer learning
        if freeze_base:
            logger.info("Freezing base model layers for transfer learning...")
            for param in model.features.parameters():
                param.requires_grad = False
            logger.debug("Base model layers frozen - only classifier will be trained")
        
        # Modify classifier for binary classification
        logger.info(f"Modifying classifier for {num_classes} output classes...")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=in_features,
            out_features=num_classes
        )
        logger.info(f"Classifier modified: {in_features} -> {num_classes}")
        
        # Move model to device
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def setup_training(model, learning_rate=1e-3, device='cpu'):
    """
    Setup loss function and optimizer for training.
    
    Args:
        model: PyTorch model
        learning_rate (float): Learning rate for optimizer (default: 1e-3)
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        tuple: (criterion, optimizer)
    """
    logger.info("Setting up loss function and optimizer...")
    try:
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        logger.info("Loss function: CrossEntropyLoss")
        
        # Define optimizer - only train classifier parameters
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=learning_rate
        )
        logger.info(f"Optimizer: Adam with learning rate={learning_rate}")
        logger.debug("Optimizer will only update classifier parameters")
        
        return criterion, optimizer
    
    except Exception as e:
        logger.error(f"Error setting up training: {str(e)}")
        raise


def get_device():
    """
    Get the appropriate device (GPU if available, else CPU).
    
    Returns:
        torch.device: Device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def model_summary(model):
    """
    Print model summary including total and trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        None
    """
    logger.info("Model Summary")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters: {frozen_params:,}")
    
    print(f"\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}\n")


if __name__ == "__main__":
    # Configure logging for standalone usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    device = get_device()
    model = load_pretrained_model(num_classes=2, freeze_base=True, device=device)
    criterion, optimizer = setup_training(model, learning_rate=1e-3, device=device)
    model_summary(model)
    
    print("Model setup completed successfully!")
