import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_data):
    """
    Evaluates the model on test data.

    Args:
        model (tf.keras.Model): The trained model.
        test_data (tf.data.Dataset): Test dataset.

    Returns:
        tuple: (test_loss, test_acc)
    """
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, test_acc

def get_predictions(model, test_data):
    """
    Gets predictions from the model.

    Args:
        model (tf.keras.Model): The trained model.
        test_data (tf.data.Dataset): Test dataset.

    Returns:
        tuple: (y_true, y_pred)
    """
    y_true = []
    y_pred = []

    for images, labels in test_data:
        predictions = model.predict(images)
        predictions = (predictions > 0.5).astype(int).flatten()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 14))
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
    plt.show()

    print(classification_report(y_true, y_pred, target_names=["WithMask", "WithoutMask"]))

def plot_history(history, fine_history=None):
    """
    Plots training history.

    Args:
        history (History): Training history.
        fine_history (History): Fine-tuning history.
    """
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

    plt.show()