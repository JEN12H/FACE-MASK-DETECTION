import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os


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