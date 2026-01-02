import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_datasets, preprocess_dataset, optimize_performance
from model import build_model, fine_tune_model
from train import train_model
from evaluate import evaluate_model, get_predictions, plot_confusion_matrix, plot_history
import tensorflow as tf

# Load data
train_data, test_data, val_data = load_datasets('data/Face Mask Dataset/Train', 'data/Face Mask Dataset/Test', 'data/Face Mask Dataset/Validation', zip_path='data/archive (5).zip')

print(train_data.class_names)

# Preprocess
train_ds = preprocess_dataset(train_data, augment=True)
val_ds = preprocess_dataset(val_data, augment=False)
test_ds = preprocess_dataset(test_data, augment=False)

# Optimize
train_ds = optimize_performance(train_ds)
val_ds = optimize_performance(val_ds)
test_ds = optimize_performance(test_ds)

# Build model
model, base_model = build_model()
model.summary()

# Train
history = train_model(model, train_ds, val_ds, epochs=5)

# Fine-tune
model = fine_tune_model(model, base_model)
fine_history = train_model(model, train_ds, val_ds, epochs=5)

# Evaluate
evaluate_model(model, test_ds)

y_true, y_pred = get_predictions(model, test_ds)
plot_confusion_matrix(y_true, y_pred)
plot_history(history, fine_history)

# Save model
model.save('models/mobilenetv2_mask_detector.h5')