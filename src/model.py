import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def build_model(input_shape=(224, 224, 3)):
    """
    Builds the MobileNetV2-based model for face mask detection.

    Args:
        input_shape (tuple): Input shape for the model.

    Returns:
        tf.keras.Model: Compiled model.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

def fine_tune_model(model, base_model):
    """
    Fine-tunes the model by unfreezing some layers.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        base_model (tf.keras.Model): The base model.

    Returns:
        tf.keras.Model: Fine-tuned model.
    """
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model