import torch
import torch.nn as nn
import logging
import yaml
import os
import sys
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_ingestion import load_data_pipeline
from model import load_pretrained_model, setup_training, get_device
from train import train_model, save_model, save_training_history
from evaluate import evaluate_and_visualize
from mlflow_tracker import setup_mlflow, log_parameters, end_mlflow_run

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_mlflow")


def load_config(config_path="params.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'batch_size': 32,
            'num_epochs': 2,
            'learning_rate': 0.001,
            'data_path': './data',
            'model_path': './models/face_mask_model.pth',
            'history_path': './metrics/training_history.json'
        }


def main():
    """
    Main training pipeline with MLflow integration.
    """
    try:
        # Load configuration
        config = load_config()
        logger.info("Training configuration loaded")
        
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Setup MLflow
        experiment_name = "Face Mask Detection - PyTorch"
        run_name = "training_run"
        setup_mlflow(experiment_name, run_name)
        logger.info("MLflow setup completed")
        
        # Prepare MLflow parameters
        mlflow_params = {
            'batch_size': config.get('batch_size', 32),
            'num_epochs': config.get('num_epochs', 2),
            'learning_rate': config.get('learning_rate', 0.001),
            'model_type': 'MobileNetV2',
            'freeze_base': True,
            'optimizer': 'Adam',
            'criterion': 'CrossEntropyLoss'
        }
        
        # Log parameters to MLflow
        logger.info("Logging parameters to MLflow...")
        for param_name, param_value in mlflow_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Load data
        logger.info("Loading data...")
        # Data structure: ../data/Face Mask Dataset/Face Mask Dataset/Train/WithMask, WithoutMask
        # Or: ../data/Train/WithMask, ../data/Validation/WithMask, ../data/Test/WithMask
        
        base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        logger.info(f"Base data path: {base_data_path}")
        
        # Check for nested Face Mask Dataset folder structure
        nested_path = os.path.join(base_data_path, 'Face Mask Dataset', 'Face Mask Dataset')
        flat_path = base_data_path
        
        # Determine which structure exists
        if os.path.exists(os.path.join(nested_path, 'Train')):
            dataset_root = nested_path
            logger.info(f"Using nested dataset structure: {dataset_root}")
        elif os.path.exists(os.path.join(flat_path, 'Train')):
            dataset_root = flat_path
            logger.info(f"Using flat dataset structure: {dataset_root}")
        else:
            raise FileNotFoundError(
                f"Could not find Train folder in:\n"
                f"  {nested_path}\n"
                f"  {flat_path}\n"
                f"Please check your data directory structure"
            )
        
        # Load transforms
        from data_ingestion import get_transforms, load_datasets, create_data_loaders
        train_transform, val_and_test_transforms = get_transforms()
        
        logger.info("Loading datasets...")
        train_dataset, val_dataset, test_dataset = load_datasets(
            dataset_root, train_transform, val_and_test_transforms
        )
        
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, 
            batch_size=config.get('batch_size', 32)
        )
        
        logger.info("Data loaded successfully")
        
        # Log dataset info
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))
        
        # Load model
        logger.info("Loading model...")
        model = load_pretrained_model(num_classes=2, freeze_base=True)
        model = model.to(device)
        logger.info("Model loaded successfully")
        
        # Log model architecture
        logger.info("Model architecture:")
        print(model)
        
        # Setup training
        logger.info("Setting up training...")
        criterion, optimizer = setup_training(
            model,
            learning_rate=config.get('learning_rate', 0.001)
        )
        logger.info("Training setup completed")
        
        # Train model with MLflow tracking
        logger.info("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config.get('num_epochs', 2),
            device=device,
            use_mlflow=True,
            mlflow_params=mlflow_params
        )
        logger.info("Training completed")
        
        # Save model
        model_path = config.get('model_path', './models/face_mask_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(model, model_path)
        
        # Log model to MLflow
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(model, "face_mask_model")
        
        # Save training history
        history_path = config.get('history_path', './metrics/training_history.json')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        save_training_history(history, history_path)
        
        # Log training history artifact
        mlflow.log_artifact(history_path, artifact_path="metrics")
        logger.info("Training history logged")
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_dir = config.get('eval_dir', './metrics/evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        evaluation_summary = evaluate_and_visualize(
            model=model,
            test_loader=test_loader,
            test_dataset=test_dataset,
            device=device,
            history=history,
            save_dir=eval_dir,
            use_mlflow=True
        )
        logger.info("Evaluation completed")
        
        # Log final summary
        logger.info("Final Evaluation Summary:")
        logger.info(f"Test Accuracy: {evaluation_summary['test_accuracy']:.4f}")
        logger.info(f"Overall Precision: {evaluation_summary['overall_precision']:.4f}")
        logger.info(f"Overall Recall: {evaluation_summary['overall_recall']:.4f}")
        logger.info(f"Overall F1-Score: {evaluation_summary['overall_f1_score']:.4f}")
        
        # End MLflow run
        logger.info("Ending MLflow run...")
        end_mlflow_run(status="FINISHED")
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"MLflow UI: Run 'mlflow ui' and visit http://localhost:5000 to view results")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        end_mlflow_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
