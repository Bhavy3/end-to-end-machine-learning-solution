import pandas as pd
import numpy as np
from data_processor import DataProcessor
from ml_models import MLModels
from sklearn.metrics import accuracy_score, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_accuracy(dataset_path, target_col, model_type, problem_type):
    # Load and process data
    df = pd.read_csv(dataset_path)
    processor = DataProcessor()
    processed_df, _ = processor.process_data(df, target_col)
    
    # Prepare features and target
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    # Train model
    ml = MLModels()
    if problem_type == 'classification':
        model, metrics = ml.train_classification(X, y, model_type)
        accuracy = metrics['accuracy'] * 100
    else:
        model, metrics = ml.train_regression(X, y, model_type)
        accuracy = metrics['r2'] * 100
    
    return accuracy, metrics

# Test simple dataset (Binary Classification)
print("\n=== Simple Dataset (Binary Classification) ===")
simple_models = ['logistic', 'rf', 'gb', 'xgboost']
for model in simple_models:
    acc, metrics = test_model_accuracy('test_simple.csv', 'target', model, 'classification')
    print(f"{model.upper()}: {acc:.2f}%")

# Test medium dataset (Multi-class Classification)
print("\n=== Medium Dataset (Multi-class Classification) ===")
medium_models = ['rf', 'gb', 'xgboost', 'catboost']
for model in medium_models:
    acc, metrics = test_model_accuracy('test_medium.csv', 'target', model, 'classification')
    print(f"{model.upper()}: {acc:.2f}%")

# Test complex dataset (Regression)
print("\n=== Complex Dataset (Regression) ===")
complex_models = ['linear', 'rf', 'gb', 'xgboost', 'catboost']
for model in complex_models:
    acc, metrics = test_model_accuracy('test_complex.csv', 'target', model, 'regression')
    print(f"{model.upper()}: {acc:.2f}%") 