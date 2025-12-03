import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import logging

logger = logging.getLogger(__name__)

# used to evaluate the model and save the evaluation
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
    
    @staticmethod
    def save_metrics(metrics: dict, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {filepath}")