import joblib
import numpy as np
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)

# used to predict the salary based on experience
class SalaryPredictor:
    def __init__(self, model_path: str):

        # loading the model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def predict(self, years_experience: Union[float, list, np.ndarray]) -> np.ndarray:
        """Make salary predictions"""
        if isinstance(years_experience, (int, float)):
            years_experience = [[years_experience]]
        elif isinstance(years_experience, list):
            years_experience = np.array(years_experience).reshape(-1, 1)
        
        predictions = self.model.predict(years_experience)
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for a batch of data"""
        predictions = self.predict(df['YearsExperience'].values)
        result_df = df.copy()
        result_df['PredictedSalary'] = predictions
        return result_df
    