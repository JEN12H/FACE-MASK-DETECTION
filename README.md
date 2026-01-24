# Face Mask Detection

A comprehensive machine learning project for detecting whether a person is wearing a face mask using Convolutional Neural Networks (CNN) based on MobileNetV2 architecture. The project includes implementations in both **TensorFlow/Keras** and **PyTorch**, a web application built with Streamlit for easy image classification, along with robust training pipelines, data handling, and model evaluation tools.

## Features

- **Dual Framework Implementation**: Both TensorFlow/Keras and PyTorch implementations for model training and inference.
- **Real-time Detection**: Upload images to classify whether a person is wearing a mask or not.
- **Pre-trained Models**: Fine-tuned MobileNetV2 models in both TensorFlow and PyTorch for accurate predictions.
- **Web Interface**: User-friendly Streamlit app for easy interaction with real-time predictions.
- **Training Pipelines**: Comprehensive training scripts with data loading, preprocessing, model training, and evaluation in both frameworks.
- **Dataset Handling**: Supports loading and preprocessing of custom datasets with data augmentation.
- **MLflow Tracking**: Experiment tracking and model registry using MLflow for PyTorch models.
- **DVC Integration**: Data and model versioning using DVC for reproducibility.
- **Configurable Parameters**: YAML-based configuration for easy model and training parameter management.
- **Logging & Monitoring**: Detailed logging and metrics tracking for model training and evaluation.
- **Model Evaluation**: Confusion matrix, classification reports, and training history visualization.
- **Inference Tools**: Prediction utilities for both frameworks.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install dependencies for TensorFlow implementation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dependencies for PyTorch implementation**:
   ```bash
   pip install -r PYTORCH/requirements.txt
   ```

4. **Download the dataset** (if not already present):
   - Download from Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
   - Extract and place the "Face Mask Dataset" folder in the `data/` directory with Train, Test, and Validation subfolders.
   - Ensure the directory structure matches:
     ```
     data/
     └── Face Mask Dataset/
         ├── Train/
         │   ├── WithMask/
         │   └── WithoutMask/
         ├── Test/
         │   ├── WithMask/
         │   └── WithoutMask/
         └── Validation/
             ├── WithMask/
             └── WithoutMask/
     ```

5. **Download the pre-trained models** (if not already present):
   - TensorFlow model: `mobilenetv2_mask_detector.h5` in the `models/` directory.
   - PyTorch model: `face_mask_model.pth` in the `PYTORCH/models/` directory.

## Usage

### Running the Web App

To launch the Streamlit web application:

- **Using the batch file** (Windows):
  ```bash
  run_app.bat
  ```

- **Manually**:
  ```bash
  streamlit run app/app.py
  ```

Open your browser and go to `http://localhost:8501` to use the app. Upload an image to get predictions.

### Training Models

#### TensorFlow/Keras Implementation

To train the TensorFlow model from scratch:

```bash
python train_model.py
```

This will:
1. Load the dataset from configured paths in `params.yaml`
2. Preprocess and augment data
3. Build the MobileNetV2 model with custom layers
4. Train the model for initial epochs
5. Fine-tune the base model
6. Evaluate on test data
7. Save the trained model in `models/` directory
8. Generate metrics, plots, and confusion matrix

**Training Configuration**: Modify `params.yaml` to adjust:
- Image size, batch size, and dataset paths
- Model architecture and layers
- Training epochs, learning rates, and optimizer settings
- Fine-tuning parameters
- Confidence thresholds for predictions

#### PyTorch Implementation

To train the PyTorch model with MLflow tracking:

```bash
cd PYTORCH
python train_with_mlflow.py
```

This will:
1. Load the dataset with PyTorch DataLoaders
2. Preprocess and augment data using torchvision transforms
3. Build or load the MobileNetV2 model
4. Train the model with specified parameters
5. Track experiments with MLflow
6. Evaluate on test data
7. Save the trained model in `PYTORCH/models/` directory
8. Generate metrics and confusion matrix

**View MLflow Experiments**:
```bash
cd PYTORCH
mlflow ui
```
Open `http://localhost:5000` to see experiment tracking dashboard.

### Evaluation

The model evaluation is performed automatically during training. You can also evaluate independently:

```python
from src.evaluate import evaluate_model, plot_confusion_matrix

# Load your model and datasets
model = tf.keras.models.load_model('models/mobilenetv2_mask_detector.h5')
# Then call evaluate_model(model, test_ds)
# Call plot_confusion_matrix() to visualize results
```

Evaluation outputs include:
- Accuracy, Precision, Recall, and F1-Score
- Confusion matrix visualization
- Classification report
- Training history plots

## Dataset

The project uses the "Face Mask Dataset" which includes:
- **Train**: Images for training the model.
- **Test**: Images for testing the model.
- **Validation**: Images for validating during training.

Each category has subfolders:
- `WithMask`: Images of people wearing masks.
- `WithoutMask`: Images of people not wearing masks.

## Project Structure

```
face-mask-detection/

├── data/                           # Dataset storage
│   └── Face Mask Dataset/
│       ├── Train/
│       │   ├── WithMask/
│       │   └── WithoutMask/
│       ├── Test/
│       │   ├── WithMask/
│       │   └── WithoutMask/
│       └── Validation/
│           ├── WithMask/
│           └── WithoutMask/

├── src/                            # TensorFlow/Keras implementation
│   ├── data_loader.py              # Data loading, preprocessing, and augmentation
│   ├── model.py                    # Model architecture and fine-tuning
│   ├── train.py                    # Training functions
│   ├── evaluate.py                 # Evaluation metrics and visualization
│   └── predict.py                  # Inference functions

├── PYTORCH/                        # PyTorch implementation
│   ├── src/
│   │   ├── data_ingestion.py       # Data loading and preprocessing
│   │   ├── model.py                # PyTorch model architecture
│   │   ├── train.py                # Training functions
│   │   ├── evaluate.py             # Evaluation and metrics
│   │   ├── predict.py              # Inference functions
│   │   └── mlflow_tracker.py       # MLflow experiment tracking
│   ├── models/
│   │   └── face_mask_model.pth     # Trained PyTorch model
│   ├── metrics/
│   │   └── *.json                  # PyTorch training metrics
│   ├── logs/
│   │   └── *.log                   # PyTorch training logs
│   ├── mlruns/                     # MLflow runs and artifacts
│   ├── train_with_mlflow.py        # Main PyTorch training script with MLflow
│   ├── PYTORCH_FACE_MASK_DETECTION.ipynb  # Jupyter notebook for PyTorch
│   └── requirements.txt            # PyTorch-specific dependencies

├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis

├── app/
│   └── app.py                      # Streamlit web application

├── models/                         # TensorFlow models
│   └── mobilenetv2_mask_detector.h5 # Trained TensorFlow model

├── metrics/                        # TensorFlow metrics
│   └── metrics.json                # Training metrics

├── plots/                          # TensorFlow visualizations
│   ├── training_history.json       # Training history
│   └── confusion_matrix.csv/json   # Confusion matrix data

├── logs/                           # TensorFlow logs
│   └── *.log                       # Training and inference logs

├── params.yaml                     # Configuration for TensorFlow training
├── dvc.yaml                        # DVC pipeline configuration
├── requirements.txt                # TensorFlow dependencies
├── train_model.py                  # Main TensorFlow training script
├── README.md                       # This file
└── LICENSE
```

## Configuration

The project uses `params.yaml` for centralized configuration:

### Data Configuration
- **Image Size**: 224x224 (for MobileNetV2)
- **Batch Size**: 32
- **Data Paths**: Configured for Train, Test, Validation splits

### Model Configuration
- **Architecture**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224x224x3
- **Output Type**: Binary classification (With/Without Mask)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dropout(0.5)
  - Dense(1, sigmoid)

### Training Configuration
- **Optimizer**: Adam (learning_rate: 0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Initial Epochs**: 5
- **Fine-tuning**: Supported with lower learning rate (0.00001)

### Prediction Configuration
- **Confidence Threshold**: 0.5
- **Class Mapping**: 0 = WithoutMask, 1 = WithMask

For full configuration details, see [params.yaml](params.yaml).

## Technologies Used

### Framework-Agnostic
- **Python 3.x**: Programming language
- **MobileNetV2**: Lightweight CNN architecture
- **Streamlit**: Web app framework for interactive interface
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities and metrics
- **DVC**: Data and model versioning
- **YAML**: Configuration management

### TensorFlow Implementation
- **TensorFlow/Keras**: Deep learning framework
- **TensorFlow Image**: Image preprocessing and augmentation

### PyTorch Implementation
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision models and transforms
- **MLflow**: Experiment tracking and model registry
- **PyTorch Lightning** (optional): Model training abstraction

## Model Details

### Architecture (Both Implementations)
Both models are built on **MobileNetV2**, a lightweight convolutional neural network designed for mobile and edge devices:
- Pre-trained weights from ImageNet
- 224×224×3 input shape
- Binary classification output layer
- Fine-tuned custom layers for face mask detection

### TensorFlow Implementation
- **Model Format**: `.h5` (Keras H5)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dropout(0.5)
  - Dense(1, sigmoid)

### PyTorch Implementation
- **Model Format**: `.pth` (PyTorch State Dict)
- **Custom Layers**:
  - AdaptiveAvgPool2d
  - Fully connected layers with ReLU and Dropout
  - Sigmoid output for binary classification

### Performance Optimization
- **Data Augmentation**: Applied to training set for better generalization
- **Batch Processing**: Efficient processing with batch size 32
- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Fine-tuning**: Optional layer unfreezing for domain adaptation
- **Mixed Precision Training** (PyTorch): Optional for faster training

### Output
The model outputs a probability (0-1) where:
- **< 0.5**: Person is NOT wearing a mask
- **≥ 0.5**: Person IS wearing a mask

## Experiment Tracking & Reproducibility

### DVC Pipeline (TensorFlow)

The project uses DVC for reproducible ML pipeline management:

```bash
dvc repro
```

This runs the entire pipeline defined in `dvc.yaml`:
- Data preparation
- Model training
- Metrics tracking
- Artifact versioning

### MLflow Tracking (PyTorch)

The PyTorch implementation uses MLflow for experiment tracking:

```bash
cd PYTORCH
mlflow ui
```

MLflow tracks:
- Hyperparameters (batch size, learning rate, epochs, etc.)
- Metrics (accuracy, loss, validation metrics)
- Model artifacts
- Training artifacts and logs
- Model registry for version management

## Outputs Generated

### TensorFlow Training Outputs
1. **Model**: `models/mobilenetv2_mask_detector.h5`
2. **Metrics**: `metrics/metrics.json` (accuracy, loss, etc.)
3. **Plots**: 
   - Training history: `plots/training_history.json`
   - Confusion matrix: `plots/confusion_matrix.csv` and `plots/confusion_matrix.json`
4. **Logs**: `logs/data_loader.log` with detailed training information

### PyTorch Training Outputs
1. **Model**: `PYTORCH/models/face_mask_model.pth`
2. **Metrics**: `PYTORCH/metrics/training_history.json` and evaluation metrics
3. **Confusion Matrix**: `PYTORCH/metrics/evaluation/confusion_matrix.json`
4. **MLflow Artifacts**: `PYTORCH/mlruns/` with experiment runs and model artifacts
5. **Logs**: `PYTORCH/logs/` with detailed training information

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Inspired by various open-source face detection projects.
