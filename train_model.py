import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_datasets, preprocess_dataset, optimize_performance
from model import build_model, fine_tune_model
from train import train_model
from evaluate import evaluate_model, get_predictions, plot_confusion_matrix, plot_history
import tensorflow as tf
import json
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Setup logging
logger = logging.getLogger("TrainPipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load params from params.yaml
def load_params(params_file='params.yaml'):
    """Load parameters from params.yaml"""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

params = load_params()

logger.info("Starting Face Mask Detection Pipeline")

# Load data
logger.info("Loading datasets...")
train_data, test_data, val_data = load_datasets(
    params['data']['train_dir'], 
    params['data']['test_dir'], 
    params['data']['val_dir'], 
    zip_path=params['data']['zip_path']
)

logger.info(f"Class names: {train_data.class_names}")

# Preprocess
logger.info("Preprocessing datasets...")
train_ds = preprocess_dataset(train_data, augment=True)
val_ds = preprocess_dataset(val_data, augment=False)
test_ds = preprocess_dataset(test_data, augment=False)

# Optimize
logger.info("Optimizing performance...")
train_ds = optimize_performance(train_ds)
val_ds = optimize_performance(val_ds)
test_ds = optimize_performance(test_ds)

# Build model
logger.info("Building model...")
model, base_model = build_model()
model.summary()

# Train
logger.info(f"Training model for {params['training']['epochs_initial']} epochs...")
history = train_model(model, train_ds, val_ds, epochs=params['training']['epochs_initial'])

# Fine-tune
logger.info("Fine-tuning model...")
model = fine_tune_model(model, base_model)
logger.info(f"Fine-tuning for {params['training']['epochs_finetune']} epochs...")
fine_history = train_model(model, train_ds, val_ds, epochs=params['training']['epochs_finetune'])

# Evaluate
logger.info("Evaluating model on test data...")
test_loss, test_acc = evaluate_model(model, test_ds)

logger.info("Generating predictions...")
y_true, y_pred = get_predictions(model, test_ds)

# Plot confusion matrix
logger.info("Plotting confusion matrix...")
plot_confusion_matrix(y_true, y_pred)

# Plot history
logger.info("Plotting training history...")
plot_history(history, fine_history)

# Save model
logger.info("Saving model...")
model_path = params['output']['model_path']
model.save(model_path)
logger.info(f"Model saved to {model_path}")

# Save metrics for DVC
logger.info("Saving metrics...")
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "training_epochs": params['training']['epochs_initial'],
    "finetuning_epochs": params['training']['epochs_finetune']
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save training history
logger.info("Saving training history...")
history_data = {
    "initial_training": {
        "accuracy": [float(x) for x in history.history['accuracy']],
        "val_accuracy": [float(x) for x in history.history['val_accuracy']],
        "loss": [float(x) for x in history.history['loss']],
        "val_loss": [float(x) for x in history.history['val_loss']]
    },
    "finetuning": {
        "accuracy": [float(x) for x in fine_history.history['accuracy']],
        "val_accuracy": [float(x) for x in fine_history.history['val_accuracy']],
        "loss": [float(x) for x in fine_history.history['loss']],
        "val_loss": [float(x) for x in fine_history.history['val_loss']]
    }
}

with open('plots/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=4)

# Save confusion matrix data
logger.info("Saving confusion matrix data...")
cm = confusion_matrix(y_true, y_pred)
cm_data = {
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_true, y_pred, target_names=["WithMask", "WithoutMask"], output_dict=True)
}

with open('plots/confusion_matrix.json', 'w') as f:
    json.dump(cm_data, f, indent=4)

# Save confusion matrix as CSV for DVC plots
import csv
with open('plots/confusion_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['actual', 'predicted', 'count'])
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            writer.writerow([i, j, cm[i][j]])

logger.info("Pipeline completed successfully!")
logger.info(f"Test Accuracy: {test_acc:.4f}")
logger.info(f"Test Loss: {test_loss:.4f}")
