import yaml
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.data_loader import DataLoader
from src.model.train import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Step 1: Load and validate data
    logger.info("Step 1: Loading and validating data...")
    loader = DataLoader(config)
    data = loader.load_data()
    if not loader.validate_data(data):
        logger.error("Data validation failed!")
        sys.exit(1)
    
    # Step 2: Split data
    logger.info("Step 2: Splitting data...")
    train_data, test_data = loader.split_data(data)
    
    # Step 3: Train model
    logger.info("Step 3: Training model...")
    trainer = ModelTrainer(config, train_data, test_data)
    
    # Train multiple models
    results = {}
    for model_name in ['linear_regression', 'random_forest']:
        logger.info(f"Training {model_name}...")
        results[model_name] = trainer.train_model(model_name)
    
    # Step 4: Select best model
    logger.info("Step 4: Selecting best model...")
    best_model = trainer.select_best_model(results)
    
    logger.info(f"Pipeline completed. Best model: {best_model}")


if __name__ == "__main__":
    main()