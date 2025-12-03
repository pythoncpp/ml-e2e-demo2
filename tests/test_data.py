import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing.data_loader import DataLoader

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'YearsExperience': [1.1, 2.0, 3.5],
        'Salary': [39343, 43525, 57189]
    })

def test_data_loader_validation(sample_data):
    config = {
        'data': {'raw_data_path': 'test.csv'},
        'model': {'test_size': 0.2, 'random_state': 42}
    }
    loader = DataLoader(config)
    
    # Test validation
    assert loader.validate_data(sample_data) == True
    
    # Test invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Salary'] = -100
    assert loader.validate_data(invalid_data) == False


def test_data_split():
    config = {
        'data': {
            'raw_data_path': 'test.csv',
            'train_data_path': 'train_test.csv',
            'test_data_path': 'test_test.csv'
        },
        'model': {'test_size': 0.2, 'random_state': 42}
    }
    
    df = pd.DataFrame({
        'YearsExperience': list(range(20)),
        'Salary': list(range(40000, 60000, 1000))
    })
    
    loader = DataLoader(config)
    train_df, test_df = loader.split_data(df)
    
    assert len(train_df) == 16  # 80% of 20
    assert len(test_df) == 4    # 20% of 20