import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import json
import csv


def setup_logging(logger_name="Evaluator"):
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
    log_file_path = os.path.join(logs_dir, 'evaluate.log')
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
logger = setup_logging("Evaluator")

def evaluate_model(model, test_data):
    """
    Evaluates the model on test data and logs performance metrics.

    Args:
        model (tf.keras.Model): The trained model.
        test_data (tf.data.Dataset): Test dataset.

    Returns:
        tuple: (test_loss, test_acc)
        
    Raises:
        Exception: If evaluation fails.
    """
    try:
        logger.info("Starting model evaluation on test data")
        
        test_loss, test_acc = model.evaluate(test_data, verbose=0)
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.debug(f"Evaluation completed - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
        raise

def get_predictions(model, test_data):
    """
    Gets predictions from the model on test data.

    Args:
        model (tf.keras.Model): The trained model.
        test_data (tf.data.Dataset): Test dataset.

    Returns:
        tuple: (y_true, y_pred) - Lists of true and predicted labels.
        
    Raises:
        Exception: If prediction generation fails.
    """
    try:
        logger.info("Generating predictions on test data")
        
        y_true = []
        y_pred = []
        batch_count = 0

        for images, labels in test_data:
            batch_count += 1
            logger.debug(f"Processing batch {batch_count}")
            
            predictions = model.predict(images, verbose=0)
            predictions = (predictions > 0.5).astype(int).flatten()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions)

        logger.info(f"Prediction generation completed - {batch_count} batches processed")
        logger.info(f"Total samples: {len(y_true)}")
        
        return y_true, y_pred
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
        raise

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots and displays the confusion matrix with classification report.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        
    Raises:
        Exception: If plotting fails.
    """
    try:
        logger.info("Generating confusion matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        logger.debug(f"Confusion matrix:\n{cm}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap="Blues",
            xticklabels=["With Mask", "Without Mask"],
            yticklabels=["With Mask", "Without Mask"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        logger.info("Confusion matrix plot displayed")
        plt.show()

        logger.info("Generating classification report")
        report = classification_report(y_true, y_pred, target_names=["WithMask", "WithoutMask"])
        logger.info(f"Classification Report:\n{report}")
        print(report)
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}", exc_info=True)
        raise

def plot_history(history, fine_history=None):
    """
    Plots training and validation history for accuracy and loss.

    Args:
        history (tf.keras.callbacks.History): Training history.
        fine_history (tf.keras.callbacks.History, optional): Fine-tuning history.
        
    Raises:
        Exception: If plotting fails.
    """
    try:
        logger.info("Generating training history plots")
        logger.debug(f"Training epochs: {len(history.history['accuracy'])}")
        
        if fine_history:
            logger.debug(f"Fine-tuning epochs: {len(fine_history.history['accuracy'])}")
        
        plt.figure(figsize=(12, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train")
        plt.plot(history.history["val_accuracy"], label="Val")
        if fine_history:
            plt.plot(
                range(len(history.history["accuracy"]),
                      len(history.history["accuracy"]) + len(fine_history.history["accuracy"])),
                fine_history.history["accuracy"],
                label="Fine-tune Train"
            )
            plt.plot(
                range(len(history.history["val_accuracy"]),
                      len(history.history["val_accuracy"]) + len(fine_history.history["val_accuracy"])),
                fine_history.history["val_accuracy"],
                label="Fine-tune Val"
            )
        plt.title("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Val")
        if fine_history:
            plt.plot(
                range(len(history.history["loss"]),
                      len(history.history["loss"]) + len(fine_history.history["loss"])),
                fine_history.history["loss"],
                label="Fine-tune Train"
            )
            plt.plot(
                range(len(history.history["val_loss"]),
                      len(history.history["val_loss"]) + len(fine_history.history["val_loss"])),
                fine_history.history["val_loss"],
                label="Fine-tune Val"
            )
        plt.title("Loss")
        plt.legend()

        logger.info("Training history plots displayed")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting history: {str(e)}", exc_info=True)
        raise


def save_metrics(test_loss, test_acc, y_true, y_pred, metrics_file='metrics.json'):
    """
    Saves evaluation metrics to a JSON file for DVC tracking.

    Args:
        test_loss (float): Test loss value.
        test_acc (float): Test accuracy value.
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        metrics_file (str): Path to save metrics JSON file. Default is 'metrics.json'.
        
    Raises:
        Exception: If saving metrics fails.
    """
    try:
        logger.info(f"Saving metrics to {metrics_file}")
        
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.dirname(metrics_file) if os.path.dirname(metrics_file) else '.'
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Calculate additional metrics
        cm = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=["WithMask", "WithoutMask"], output_dict=True)
        
        # True Negatives, False Positives, False Negatives, True Positives
        tn, fp, fn, tp = cm.ravel()
        
        metrics_data = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "total_samples": len(y_true),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "precision": float(class_report["weighted avg"]["precision"]),
            "recall": float(class_report["weighted avg"]["recall"]),
            "f1_score": float(class_report["weighted avg"]["f1-score"]),
            "class_metrics": {
                "WithMask": {
                    "precision": float(class_report["WithMask"]["precision"]),
                    "recall": float(class_report["WithMask"]["recall"]),
                    "f1_score": float(class_report["WithMask"]["f1-score"]),
                    "support": int(class_report["WithMask"]["support"])
                },
                "WithoutMask": {
                    "precision": float(class_report["WithoutMask"]["precision"]),
                    "recall": float(class_report["WithoutMask"]["recall"]),
                    "f1_score": float(class_report["WithoutMask"]["f1-score"]),
                    "support": int(class_report["WithoutMask"]["support"])
                }
            }
        }
        
        # Save to JSON file
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        logger.info(f"Metrics saved successfully to {metrics_file}")
        logger.debug(f"Metrics content: {json.dumps(metrics_data, indent=2)}")
        
        return metrics_data
        
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}", exc_info=True)
        raise


def save_confusion_matrix_csv(y_true, y_pred, csv_file='plots/confusion_matrix.csv'):
    """
    Saves confusion matrix data to CSV for DVC plots.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        csv_file (str): Path to save confusion matrix CSV file. Default is 'plots/confusion_matrix.csv'.
        
    Raises:
        Exception: If saving CSV fails.
    """
    try:
        logger.info(f"Saving confusion matrix CSV to {csv_file}")
        
        # Create plots directory if it doesn't exist
        csv_dir = os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.'
        os.makedirs(csv_dir, exist_ok=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['actual', 'predicted', 'count'])
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    writer.writerow([i, j, cm[i][j]])
        
        logger.info(f"Confusion matrix CSV saved successfully to {csv_file}")
        
    except Exception as e:
        logger.error(f"Error saving confusion matrix CSV: {str(e)}", exc_info=True)
        raise


def save_training_history(history, fine_history=None, history_file='plots/training_history.json'):
    """
    Saves training and fine-tuning history to JSON file.

    Args:
        history (tf.keras.callbacks.History): Training history.
        fine_history (tf.keras.callbacks.History, optional): Fine-tuning history.
        history_file (str): Path to save history JSON file. Default is 'plots/training_history.json'.
        
    Raises:
        Exception: If saving history fails.
    """
    try:
        logger.info(f"Saving training history to {history_file}")
        
        # Create plots directory if it doesn't exist
        history_dir = os.path.dirname(history_file) if os.path.dirname(history_file) else '.'
        os.makedirs(history_dir, exist_ok=True)
        
        history_data = {
            "initial_training": {
                "accuracy": [float(x) for x in history.history['accuracy']],
                "val_accuracy": [float(x) for x in history.history['val_accuracy']],
                "loss": [float(x) for x in history.history['loss']],
                "val_loss": [float(x) for x in history.history['val_loss']]
            }
        }
        
        if fine_history:
            history_data["finetuning"] = {
                "accuracy": [float(x) for x in fine_history.history['accuracy']],
                "val_accuracy": [float(x) for x in fine_history.history['val_accuracy']],
                "loss": [float(x) for x in fine_history.history['loss']],
                "val_loss": [float(x) for x in fine_history.history['val_loss']]
            }
            logger.debug(f"Saved initial training and fine-tuning history")
        else:
            logger.debug(f"Saved initial training history only")
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        logger.info(f"Training history saved successfully to {history_file}")
        
    except Exception as e:
        logger.error(f"Error saving training history: {str(e)}", exc_info=True)
        raise