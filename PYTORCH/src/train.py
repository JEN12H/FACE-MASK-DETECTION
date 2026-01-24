"""
Face Mask Detection - PyTorch Model Training Module

This module handles model training and validation with MLflow tracking.
For predictions and evaluation, see predict.py module.
"""

import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import json
import mlflow

# Setup logging
logger = logging.getLogger("train")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    logger.debug("Starting training epoch...")
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_train_samples = 0

    try:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_train_samples += images.size(0)

        epoch_loss = running_loss / total_train_samples
        epoch_accuracy = correct_predictions / total_train_samples
        
        logger.debug(f"Training epoch complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        return epoch_loss, epoch_accuracy
    
    except Exception as e:
        logger.error(f"Error during training epoch: {str(e)}")
        raise


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on ('cpu' or 'cuda')
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    logger.debug("Starting validation epoch...")
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    total_val_samples = 0

    try:
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                total_val_samples += images.size(0)

        epoch_loss = val_running_loss / total_val_samples
        epoch_accuracy = val_correct_predictions / total_val_samples
        
        logger.debug(f"Validation epoch complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        return epoch_loss, epoch_accuracy
    
    except Exception as e:
        logger.error(f"Error during validation epoch: {str(e)}")
        raise


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, use_mlflow=True, mlflow_params=None):
    """
    Train the model for multiple epochs with validation and optional MLflow tracking.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of epochs to train
        device: Device to train on ('cpu' or 'cuda')
        use_mlflow (bool): Whether to use MLflow tracking (default: True)
        mlflow_params (dict): Parameters to log to MLflow
    
    Returns:
        dict: Dictionary containing training history
    """
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    if use_mlflow and mlflow_params:
        logger.info("MLflow tracking enabled")
        for param_name, param_value in mlflow_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("device", str(device))
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    try:
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Log to MLflow
            if use_mlflow:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            # Log epoch results
            logger.info(
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
            )
            print(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
            )
        
        logger.info("Training completed successfully!")
        
        # Return training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
        
        return history
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise





def save_model(model, model_path):
    """
    Save model state dictionary.
    
    Args:
        model: PyTorch model
        model_path (str): Path to save model
    
    Returns:
        None
    """
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved successfully to: {model_path}")
        print(f"Model saved successfully to: {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def save_training_history(history, history_path):
    """
    Save training history to JSON file.
    
    Args:
        history (dict): Training history dictionary
        history_path (str): Path to save history
    
    Returns:
        None
    """
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Training history saved to: {history_path}")
        print(f"Training history saved to: {history_path}")
    except Exception as e:
        logger.error(f"Error saving training history: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Training module loaded successfully!")
