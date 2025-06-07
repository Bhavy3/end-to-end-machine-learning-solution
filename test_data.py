import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data():
    """Generate test datasets for different scenarios"""
    
    # 1. Classification Dataset
    np.random.seed(42)
    n_samples = 1000
    
    classification_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # 2. Regression Dataset
    regression_data = pd.DataFrame({
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.normal(20, 10, n_samples),
        'location_score': np.random.normal(7, 2, n_samples),
        'price': np.random.normal(300000, 100000, n_samples)
    })
    
    # 3. Time Series Dataset
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    time_series_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(100, 20, 365) + 
                np.sin(np.arange(365) * 2 * np.pi / 365) * 30 +  # Yearly seasonality
                np.sin(np.arange(365) * 2 * np.pi / 7) * 10 +    # Weekly seasonality
                np.arange(365) * 0.1                             # Trend
    })
    
    # Save datasets
    classification_data.to_csv('test_classification.csv', index=False)
    regression_data.to_csv('test_regression.csv', index=False)
    time_series_data.to_csv('test_timeseries.csv', index=False)
    
    print("✅ Test datasets generated:")
    print("1. test_classification.csv - For classification tasks")
    print("2. test_regression.csv - For regression tasks")
    print("3. test_timeseries.csv - For time series analysis")

if __name__ == "__main__":
    generate_test_data() 