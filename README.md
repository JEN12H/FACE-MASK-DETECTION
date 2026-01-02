# Face Mask Detection

A machine learning project for detecting whether a person is wearing a face mask using a Convolutional Neural Network (CNN) based on MobileNetV2 architecture. The project includes a web application built with Streamlit for easy image classification.

## Features

- **Real-time Detection**: Upload images to classify whether a person is wearing a mask or not.
- **Pre-trained Model**: Uses a fine-tuned MobileNetV2 model for accurate predictions.
- **Web Interface**: User-friendly Streamlit app for easy interaction.
- **Training Scripts**: Includes scripts for data loading, model training, and evaluation.
- **Dataset Handling**: Supports loading and preprocessing of custom datasets.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (if not already present):
   - Place the "Face Mask Dataset" folder in the `data/` directory with Train, Test, and Validation subfolders.

4. **Download the pre-trained model** (if not already present):
   - The model `mobilenetv2_mask_detector.h5` should be in the `models/` directory.

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

### Training the Model

To train the model from scratch:

```bash
python train_model.py
```

This will load the dataset, preprocess it, build the model, and train it. The trained model will be saved in the `models/` directory.

### Evaluation

To evaluate the trained model:

```python
from src.evaluate import evaluate_model, plot_confusion_matrix

# Load your model and datasets
# Then call evaluate_model(model, test_ds)
```

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
│
├── data/
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
│
├── notebooks/
│   └── eda.ipynb  # Exploratory Data Analysis notebook
│
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # Model building and fine-tuning
│   ├── train.py          # Training functions
│   ├── evaluate.py       # Evaluation and plotting
│   └── predict.py        # Prediction functions
│
├── app/
│   └── app.py            # Streamlit web application
│
├── models/
│   └── mobilenetv2_mask_detector.h5  # Pre-trained model
│
├── requirements.txt      # Python dependencies
├── run_app.bat           # Batch file to run the app
├── train_model.py        # Script to train the model
├── README.md             # This file
└── LICENSE
```

## Technologies Used

- **Python**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Base model architecture
- **Streamlit**: Web app framework
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities

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
