import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

# used to train the model
class ModelTrainer:
    def __init__(self, config: dict, train_data, test_data):
        self.config = config
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        # set the data
        self.train_df = train_data
        self.test_df = test_data     
    
    def load_data(self) -> tuple:
        """Load training data"""
        
        # split the data into X and y
        X_train = self.train_df[['YearsExperience']].values
        y_train = self.train_df['Salary'].values
        X_test = self.test_df[['YearsExperience']].values
        y_test = self.test_df['Salary'].values
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, model_name: str = 'linear_regression'):
        """Train model with MLflow tracking"""

        # read the data
        X_train, y_train, X_test, y_test = self.load_data()

        # get the model from the model list
        model = self.models[model_name]
    
        with mlflow.start_run(run_name=f"{model_name}_run"):
            # Log parameters
            if model_name == 'random_forest':
                mlflow.log_param("n_estimators", 100)
            
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, name="model")

            # Save model locally
            model_path = os.path.join(
                self.config['model']['model_dir'],
                f"{self.config['model']['model_name']}_{model_name}.pkl"
            )
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            logger.info(f"{model_name} trained. MSE: {mse:.2f}, R2: {r2:.2f}")
            
            return {
                'model': model,
                'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
                'predictions': y_pred
            }
    
    def select_best_model(self, results: Dict[str, Dict]) -> str:
        """Select best model based on R2 score"""
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['r2'])
        logger.info(f"Best model: {best_model[0]} with R2: {best_model[1]['metrics']['r2']:.2f}")
        return best_model[0]
            
            
        