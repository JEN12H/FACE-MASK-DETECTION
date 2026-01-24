"""
Face Mask Detection - PyTorch Model Evaluation and Visualization Module

This module handles detailed model evaluation, visualization, analysis, and MLflow logging.
Includes confusion matrix, classification report, and training history plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import mlflow
import os

# Setup logging
logger = logging.getLogger("evaluate")


def plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=None):
    """
    Plot and display confusion matrix.
    
    Args:
        all_labels (np.array): True labels
        all_predictions (np.array): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the plot (optional)
    
    Returns:
        np.array: Confusion matrix
    """
    logger.info("Generating confusion matrix plot...")
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to: {save_path}")
        
        plt.show()
        logger.info("Confusion matrix plot displayed")
        
        return cm
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise


def plot_classification_metrics(all_labels, all_predictions, class_names):
    """
    Plot classification metrics (precision, recall, F1-score) for each class.
    
    Args:
        all_labels (np.array): True labels
        all_predictions (np.array): Predicted labels
        class_names (list): List of class names
    
    Returns:
        dict: Detailed classification metrics
    """
    logger.info("Generating classification metrics visualization...")
    try:
        # Get metrics
        report_dict = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Extract metrics for each class
        classes = class_names
        precision = [report_dict[c]['precision'] for c in classes]
        recall = [report_dict[c]['recall'] for c in classes]
        f1_score = [report_dict[c]['f1-score'] for c in classes]
        
        # Plot
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Classification Metrics per Class')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        logger.info("Classification metrics plot displayed")
        return report_dict
    
    except Exception as e:
        logger.error(f"Error plotting classification metrics: {str(e)}")
        raise


def display_classification_report(all_labels, all_predictions, class_names):
    """
    Display classification report with detailed metrics.
    
    Args:
        all_labels (np.array): True labels
        all_predictions (np.array): Predicted labels
        class_names (list): List of class names
    
    Returns:
        str: Classification report
    """
    logger.info("Generating classification report...")
    try:
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            digits=4
        )
        
        logger.info("Classification report generated")
        print("=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        raise


def display_confusion_matrix(all_labels, all_predictions, class_names):
    """
    Display confusion matrix in text format.
    
    Args:
        all_labels (np.array): True labels
        all_predictions (np.array): Predicted labels
        class_names (list): List of class names
    
    Returns:
        np.array: Confusion matrix
    """
    logger.info("Generating confusion matrix...")
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        
        logger.info("Confusion matrix generated")
        print("=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        
        # Print header
        print("\nPredicted →")
        print("Actual ↓", end="")
        for name in class_names:
            print(f"\t{name}", end="")
        print("\n" + "-" * (30 + len(class_names) * 15))
        
        # Print confusion matrix
        for i, name in enumerate(class_names):
            print(f"{name}", end="")
            for j in range(len(class_names)):
                print(f"\t{cm[i][j]}", end="")
            print()
        
        print("=" * 80)
        
        return cm
    
    except Exception as e:
        logger.error(f"Error displaying confusion matrix: {str(e)}")
        raise


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.
    
    Args:
        history (dict): Training history dictionary with keys:
                       'train_losses', 'val_losses', 'train_accuracies', 'val_accuracies'
        save_path (str): Path to save the plot (optional)
    
    Returns:
        None
    """
    logger.info("Plotting training history...")
    try:
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        train_accuracies = history.get('train_accuracies', [])
        val_accuracies = history.get('val_accuracies', [])
        
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
        logger.info("Training history plot displayed")
    
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise


def generate_evaluation_summary(all_labels, all_predictions, class_names, 
                               test_accuracy, training_history=None):
    """
    Generate comprehensive evaluation summary report.
    
    Args:
        all_labels (np.array): True labels
        all_predictions (np.array): Predicted labels
        class_names (list): List of class names
        test_accuracy (float): Test accuracy
        training_history (dict): Training history (optional)
    
    Returns:
        dict: Comprehensive evaluation summary
    """
    logger.info("Generating evaluation summary...")
    try:
        # Get metrics
        cm = confusion_matrix(all_labels, all_predictions)
        report_dict = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_name in class_names:
            per_class_metrics[class_name] = {
                'precision': float(report_dict[class_name]['precision']),
                'recall': float(report_dict[class_name]['recall']),
                'f1-score': float(report_dict[class_name]['f1-score']),
                'support': int(report_dict[class_name]['support'])
            }
        
        # Compile summary
        summary = {
            'test_accuracy': float(test_accuracy),
            'overall_precision': float(report_dict['weighted avg']['precision']),
            'overall_recall': float(report_dict['weighted avg']['recall']),
            'overall_f1_score': float(report_dict['weighted avg']['f1-score']),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        if training_history:
            summary['training_history'] = training_history
        
        logger.info("Evaluation summary generated successfully")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating evaluation summary: {str(e)}")
        raise


def save_evaluation_report(summary, report_path):
    """
    Save evaluation report to JSON file.
    
    Args:
        summary (dict): Evaluation summary dictionary
        report_path (str): Path to save report
    
    Returns:
        None
    """
    try:
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Evaluation report saved to: {report_path}")
        print(f"Evaluation report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation report: {str(e)}")
        raise


def save_model(model, model_path):
    """
    Save model state dictionary to disk.
    
    Args:
        model: PyTorch model
        model_path (str): Path to save model (e.g., 'face_mask_model.pth')
    
    Returns:
        None
    """
    logger.info(f"Saving model to: {model_path}")
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved successfully to: {model_path}")
        print(f"Model saved successfully as {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model_state(model, model_path, device):
    """
    Load previously saved model state dictionary.
    
    Args:
        model: PyTorch model
        model_path (str): Path to saved model
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Model with loaded state
    """
    logger.info(f"Loading model from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded successfully from: {model_path}")
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def save_confusion_matrix_to_json(cm, class_names, save_path):
    """
    Save confusion matrix to JSON file.
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save JSON
    
    Returns:
        None
    """
    try:
        cm_dict = {
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        with open(save_path, 'w') as f:
            json.dump(cm_dict, f, indent=4)
        logger.info(f"Confusion matrix saved to: {save_path}")
        print(f"Confusion matrix saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix to JSON: {str(e)}")
        raise





def evaluate_and_visualize(model, test_loader, test_dataset, device, 
                          history=None, save_dir=None, use_mlflow=True):
    """
    Complete evaluation pipeline with all visualizations and optional MLflow logging.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        test_dataset: Test dataset
        device: Device to use
        history (dict): Training history (optional)
        save_dir (str): Directory to save plots (optional)
        use_mlflow (bool): Whether to log to MLflow (default: True)
    
    Returns:
        dict: Comprehensive evaluation results
    """
    logger.info("Starting comprehensive evaluation and visualization...")
    try:
        # Import predict functions
        from predict import test_model, get_predictions
        
        # Get test accuracy
        test_accuracy = test_model(model, test_loader, device)
        
        # Get predictions
        all_labels, all_predictions = get_predictions(model, test_loader, device)
        
        # Get class names
        class_names = test_dataset.classes
        
        # Display metrics
        display_classification_report(all_labels, all_predictions, class_names)
        display_confusion_matrix(all_labels, all_predictions, class_names)
        
        # Plot visualizations
        plot_confusion_matrix(all_labels, all_predictions, class_names)
        plot_classification_metrics(all_labels, all_predictions, class_names)
        
        if history:
            plot_training_history(history)
        
        # Generate summary
        summary = generate_evaluation_summary(
            all_labels, 
            all_predictions, 
            class_names, 
            test_accuracy, 
            history
        )
        
        # Save reports if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_evaluation_report(summary, os.path.join(save_dir, 'evaluation_report.json'))
            save_confusion_matrix_to_json(
                confusion_matrix(all_labels, all_predictions),
                class_names,
                os.path.join(save_dir, 'confusion_matrix.json')
            )
        
        # Log to MLflow
        if use_mlflow:
            logger.info("Logging evaluation metrics to MLflow...")
            
            # Log test metrics
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            # Log per-class metrics from classification report
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            
            # Log artifacts (plots and reports)
            if save_dir:
                logger.info("Logging artifacts to MLflow...")
                for filename in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, filename)
                    if os.path.isfile(file_path):
                        mlflow.log_artifact(file_path, artifact_path="evaluation")
            
            logger.info("Evaluation metrics logged to MLflow successfully")
        
        logger.info("Comprehensive evaluation and visualization completed")
        return summary
    
    except Exception as e:
        logger.error(f"Error during evaluation and visualization: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Evaluation module loaded successfully!")
