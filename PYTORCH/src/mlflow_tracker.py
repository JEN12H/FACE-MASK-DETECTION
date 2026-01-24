"""
MLflow Integration for Face Mask Detection PyTorch Project

This module provides MLflow tracking capabilities for experiments, metrics, and artifacts.
"""

import mlflow
import mlflow.pytorch
import logging
import json
import os

# Setup logging
logger = logging.getLogger("mlflow_tracking")


def setup_mlflow(experiment_name, run_name=None, tracking_uri="./mlruns"):
    """
    Setup MLflow experiment and start a new run.
    
    Args:
        experiment_name (str): Name of the experiment
        run_name (str): Name of the current run (optional)
        tracking_uri (str): URI where MLflow stores data (default: './mlruns')
    
    Returns:
        None
    """
    logger.info(f"Setting up MLflow with experiment: {experiment_name}")
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug(f"Tracking URI set to: {tracking_uri}")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
        
        # Start a new run
        if run_name:
            mlflow.start_run(run_name=run_name)
            logger.info(f"MLflow run started: {run_name}")
        else:
            mlflow.start_run()
            logger.info("MLflow run started")
    
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise


def log_parameters(params_dict):
    """
    Log training parameters to MLflow.
    
    Args:
        params_dict (dict): Dictionary of parameters to log
    
    Returns:
        None
    """
    logger.info("Logging parameters to MLflow...")
    try:
        for param_name, param_value in params_dict.items():
            # Convert complex types to strings
            if isinstance(param_value, (dict, list)):
                param_value = json.dumps(param_value)
            mlflow.log_param(param_name, param_value)
        logger.info(f"Logged {len(params_dict)} parameters")
    except Exception as e:
        logger.error(f"Error logging parameters: {str(e)}")
        raise


def log_training_metrics(epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Log training metrics for each epoch.
    
    Args:
        epoch (int): Epoch number
        train_loss (float): Training loss
        train_accuracy (float): Training accuracy
        val_loss (float): Validation loss
        val_accuracy (float): Validation accuracy
    
    Returns:
        None
    """
    logger.debug(f"Logging metrics for epoch {epoch}")
    try:
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
    except Exception as e:
        logger.error(f"Error logging training metrics: {str(e)}")
        raise


def log_test_metrics(test_accuracy, precision, recall, f1_score):
    """
    Log test evaluation metrics.
    
    Args:
        test_accuracy (float): Test accuracy
        precision (float): Weighted average precision
        recall (float): Weighted average recall
        f1_score (float): Weighted average F1-score
    
    Returns:
        None
    """
    logger.info("Logging test metrics to MLflow...")
    try:
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1_score)
        logger.info("Test metrics logged successfully")
    except Exception as e:
        logger.error(f"Error logging test metrics: {str(e)}")
        raise


def log_per_class_metrics(per_class_metrics):
    """
    Log per-class evaluation metrics.
    
    Args:
        per_class_metrics (dict): Dictionary of per-class metrics
    
    Returns:
        None
    """
    logger.info("Logging per-class metrics to MLflow...")
    try:
        for class_name, metrics in per_class_metrics.items():
            for metric_name, metric_value in metrics.items():
                if metric_name != 'support':  # Skip support count
                    full_metric_name = f"{class_name}_{metric_name}"
                    mlflow.log_metric(full_metric_name, metric_value)
        logger.info(f"Logged per-class metrics for {len(per_class_metrics)} classes")
    except Exception as e:
        logger.error(f"Error logging per-class metrics: {str(e)}")
        raise


def log_confusion_matrix(cm, class_names):
    """
    Log confusion matrix as an artifact.
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
    
    Returns:
        None
    """
    logger.info("Logging confusion matrix to MLflow...")
    try:
        cm_dict = {
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        # Save to temporary file
        temp_path = "cm_temp.json"
        with open(temp_path, 'w') as f:
            json.dump(cm_dict, f, indent=4)
        
        # Log as artifact
        mlflow.log_artifact(temp_path, artifact_path="matrices")
        
        # Clean up
        os.remove(temp_path)
        logger.info("Confusion matrix logged successfully")
    
    except Exception as e:
        logger.error(f"Error logging confusion matrix: {str(e)}")
        raise


def log_training_history(history):
    """
    Log training history as an artifact.
    
    Args:
        history (dict): Training history dictionary
    
    Returns:
        None
    """
    logger.info("Logging training history to MLflow...")
    try:
        # Save to temporary file
        temp_path = "history_temp.json"
        with open(temp_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Log as artifact
        mlflow.log_artifact(temp_path, artifact_path="history")
        
        # Clean up
        os.remove(temp_path)
        logger.info("Training history logged successfully")
    
    except Exception as e:
        logger.error(f"Error logging training history: {str(e)}")
        raise


def log_model(model, model_name="face_mask_model"):
    """
    Log PyTorch model to MLflow.
    
    Args:
        model: PyTorch model
        model_name (str): Name of the model
    
    Returns:
        None
    """
    logger.info(f"Logging PyTorch model to MLflow: {model_name}")
    try:
        mlflow.pytorch.log_model(model, model_name)
        logger.info("Model logged successfully")
    except Exception as e:
        logger.error(f"Error logging model: {str(e)}")
        raise


def log_evaluation_report(evaluation_summary, report_name="evaluation_report"):
    """
    Log complete evaluation report as artifact.
    
    Args:
        evaluation_summary (dict): Evaluation summary dictionary
        report_name (str): Name of the report
    
    Returns:
        None
    """
    logger.info("Logging evaluation report to MLflow...")
    try:
        # Save to temporary file
        temp_path = f"{report_name}_temp.json"
        with open(temp_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=4)
        
        # Log as artifact
        mlflow.log_artifact(temp_path, artifact_path="reports")
        
        # Clean up
        os.remove(temp_path)
        logger.info("Evaluation report logged successfully")
    
    except Exception as e:
        logger.error(f"Error logging evaluation report: {str(e)}")
        raise


def end_mlflow_run(status="FINISHED"):
    """
    End the current MLflow run.
    
    Args:
        status (str): Status of the run (FINISHED, FAILED, etc.)
    
    Returns:
        None
    """
    logger.info(f"Ending MLflow run with status: {status}")
    try:
        mlflow.end_run(status=status)
        logger.info("MLflow run ended successfully")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {str(e)}")
        raise


def get_best_run(experiment_name, metric_name="test_accuracy"):
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name (str): Name of the experiment
        metric_name (str): Metric to use for comparison (default: 'test_accuracy')
    
    Returns:
        dict: Best run information
    """
    logger.info(f"Getting best run from experiment: {experiment_name}")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"Experiment {experiment_name} not found")
            return None
        
        # Get all runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning(f"No runs found for experiment {experiment_name}")
            return None
        
        # Get best run based on metric
        best_run = runs.loc[runs[f'metrics.{metric_name}'].idxmax()]
        logger.info(f"Best run found: {best_run['run_id']}")
        
        return best_run
    
    except Exception as e:
        logger.error(f"Error getting best run: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("MLflow tracking module loaded successfully!")
