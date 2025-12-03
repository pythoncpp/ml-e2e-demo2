import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# used for loading and preparing the data
class DataLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(self.config['data']['raw_data_path'])
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        # split the data into X and y
        X = df[['YearsExperience']]
        y = df['Salary']
        
        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state']
        )
        
        # create a single data frame with both training X and y
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save split data
        train_df.to_csv(self.config['data']['train_data_path'], index=False)
        test_df.to_csv(self.config['data']['test_data_path'], index=False)
        
        logger.info(f"Data split completed. Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data schema and quality"""
        required_columns = ['YearsExperience', 'Salary']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
        
        # Check for null values
        if df.isnull().sum().any():
            logger.error("Data contains null values")
            return False
        
        # Check for negative values
        if (df['YearsExperience'] < 0).any() or (df['Salary'] < 0).any():
            logger.error("Data contains negative values")
            return False
        
        logger.info("Data validation passed")
        return True