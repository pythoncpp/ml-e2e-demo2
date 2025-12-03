import pytest
import numpy as np
from src.model.train import ModelTrainer
import os

def test_model_training():
    config = {
        'data': {
            'train_data_path': 'test_train.csv',
            'test_data_path': 'test_test.csv'
        },
        'model': {
            'model_dir': 'test_models/',
            'model_name': 'test_model',
            'test_size': 0.2,
            'random_state': 42
        },
        'mlflow': {
            'tracking_uri': 'http://localhost:8000',
            'experiment_name': 'test_experiment'
        }
    }

    # create a directory
    # os.mkdir('../test_models')   
    
    # Create test data
    np.random.seed(42)
    X_train = np.random.rand(100, 1) * 10
    y_train = 50000 + 10000 * X_train.flatten() + np.random.randn(100) * 5000
    
    X_test = np.random.rand(20, 1) * 10
    y_test = 50000 + 10000 * X_test.flatten() + np.random.randn(20) * 5000
    
    # Save test data
    import pandas as pd
    train_df = pd.DataFrame({'YearsExperience': X_train.flatten(), 'Salary': y_train})
    test_df = pd.DataFrame({'YearsExperience': X_test.flatten(), 'Salary': y_test})
    
    train_df.to_csv('test_train.csv', index=False)
    test_df.to_csv('test_test.csv', index=False)
    
    # Test training
    trainer = ModelTrainer(config, train_df, test_df)
    results = trainer.train_model('linear_regression')
    
    assert 'model' in results
    assert 'metrics' in results
    assert 'predictions' in results
    assert results['metrics']['r2'] > 0.8  # Should have decent fit
    
    # Cleanup
    os.remove('test_train.csv')
    os.remove('test_test.csv')