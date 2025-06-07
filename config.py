import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Data Processing Settings
DEFAULT_SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust'
DEFAULT_IMPUTATION_STRATEGY = 'mean'  # Options: 'mean', 'median', 'most_frequent'
CLASSIFICATION_THRESHOLD = 10  # Number of unique values to consider for classification

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Advanced Model Settings
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror'
}

LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'regression'
}

CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function': 'RMSE'
}

# Time Series Settings
PROPHET_PARAMS = {
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0
}

# Explainability Settings
SHAP_SETTINGS = {
    'max_display': 20,
    'plot_type': 'bar'
}

LIME_SETTINGS = {
    'num_features': 10,
    'num_samples': 1000
}

# API Settings
FASTAPI_SETTINGS = {
    'title': 'AI Data Science Assistant API',
    'version': '1.0.0',
    'debug': False
}

# Model Optimization Settings
OPTUNA_SETTINGS = {
    'n_trials': 100,
    'timeout': 600,
    'n_jobs': -1
}

# Anomaly Detection Settings
ANOMALY_DETECTION = {
    'contamination': 0.1,
    'random_state': 42
}

# File Paths
MODEL_SAVE_PATH = 'models/'
PLOT_SAVE_PATH = 'plots/'
DATA_SAVE_PATH = 'data/processed/' 