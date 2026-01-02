def train_model(model, train_data, validation_data, epochs=5):
    """
    Trains the model.

    Args:
        model (tf.keras.Model): The model to train.
        train_data (tf.data.Dataset): Training dataset.
        validation_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs.

    Returns:
        History: Training history.
    """
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs
    )
    return history

def fine_tune_model(model, train_data, validation_data, epochs=5):
    """
    Fine-tunes the model.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        train_data (tf.data.Dataset): Training dataset.
        validation_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs.

    Returns:
        History: Fine-tuning history.
    """
    fine_tune_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs
    )
    return fine_tune_history