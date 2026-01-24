"""
Face Mask Detection - PyTorch Prediction Module

This module handles model predictions and inference on single images.
For detailed evaluation metrics and visualizations, see evaluate.py module.
"""

import torch
import numpy as np
import logging

# Setup logging
logger = logging.getLogger("predict")


def test_model(model, test_loader, device):
    """
    Evaluate the model on test dataset and return test accuracy.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to test on ('cpu' or 'cuda')
    
    Returns:
        float: Test accuracy
    """
    logger.info("Starting test evaluation...")
    model.eval()
    test_correct_predictions = 0
    test_total = 0

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                test_total += labels.size(0)
                test_correct_predictions += (predicted == labels).sum().item()

        test_accuracy = test_correct_predictions / test_total
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        print(f'Test Accuracy: {test_accuracy:.4f}')
        
        return test_accuracy
    
    except Exception as e:
        logger.error(f"Error during test evaluation: {str(e)}")
        raise


def get_predictions(model, test_loader, device):
    """
    Collect all predictions and labels from test loader for detailed evaluation.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        tuple: (all_labels, all_predictions) as numpy arrays
    """
    logger.info("Collecting predictions for detailed evaluation...")
    
    model.eval()
    all_labels = []
    all_predictions = []

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        logger.info(f"Predictions collected: {len(all_predictions)} samples")
        logger.debug(f"Labels shape: {all_labels.shape}, Predictions shape: {all_predictions.shape}")
        return all_labels, all_predictions
    
    except Exception as e:
        logger.error(f"Error collecting predictions: {str(e)}")
        raise


def predict_single_image(model, image, device, class_names):
    """
    Make prediction on a single image.
    
    Args:
        model: PyTorch model
        image: Input image tensor (preprocessed)
        device: Device to use ('cpu' or 'cuda')
        class_names (list): List of class names
    
    Returns:
        dict: Prediction result with class and confidence
    """
    logger.debug("Making prediction on single image...")
    model.eval()
    try:
        with torch.no_grad():
            # Ensure image is on correct device
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            image = image.to(device)
            
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Get prediction
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
            
            logger.debug(f"Prediction: {predicted_class}, Confidence: {confidence_score:.4f}")
            
            result = {
                'predicted_class': predicted_class,
                'confidence': float(confidence_score),
                'probabilities': {class_names[i]: float(p) for i, p in enumerate(probabilities[0])}
            }
            
            return result
    
    except Exception as e:
        logger.error(f"Error making prediction on single image: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Prediction module loaded successfully!")
