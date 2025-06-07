import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging
import re
from difflib import get_close_matches
from config import (
    OPENAI_API_KEY, DEFAULT_SCALING_METHOD, 
    DEFAULT_IMPUTATION_STRATEGY, CLASSIFICATION_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT
)
from ml_models import MLModels

class DataProcessor:
    def __init__(self, scaling_method=DEFAULT_SCALING_METHOD):
        """
        Initialize DataProcessor with configurable scaling method
        """
        self.scaling_method = scaling_method
        self._setup_scalers()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy=DEFAULT_IMPUTATION_STRATEGY)
        self._setup_logging()
        self.ml_models = MLModels()
        
    def _setup_scalers(self):
        """Setup appropriate scaler based on configuration"""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
    def detect_problem_type(self, target_column, data):
        """
        Detect if the problem is classification or regression based on target variable
        """
        unique_values = data[target_column].nunique()
        if unique_values <= CLASSIFICATION_THRESHOLD:
            return 'classification'
        return 'regression'
    
    def validate_data(self, df):
        """
        Validate input dataframe for common issues
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            self.logger.warning("Infinite values detected in numeric columns")
            
        # Check for all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            self.logger.warning(f"Columns with all null values: {null_cols}")
            
        return True
    
    def clean_data(self, df):
        """
        Perform enhanced data cleaning operations
        """
        self.validate_data(df)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            self.logger.info(f"Imputed missing values in numeric columns: {numeric_cols.tolist()}")
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            self.logger.info(f"Imputed missing values in categorical column: {col}")
            
        return df
    
    def encode_categorical(self, df, categorical_columns):
        """
        Enhanced categorical variable encoding with better handling of different types
        """
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Handle missing values
            df[col] = df[col].fillna('MISSING')
            
            # Convert to string and handle special characters
            df[col] = df[col].astype(str).str.strip()
            
            # Handle binary categorical variables
            if df[col].nunique() == 2:
                df[col] = (df[col] == df[col].value_counts().index[0]).astype(int)
                self.logger.info(f"Binary encoded column: {col}")
            else:
                # Label encode other categorical variables
                df[col] = self.label_encoders[col].fit_transform(df[col])
                self.logger.info(f"Label encoded column: {col}")
                
                # For high cardinality categorical variables, consider one-hot encoding
                if df[col].nunique() > 10:
                    self.logger.warning(f"High cardinality in column {col}: {df[col].nunique()} unique values")
                    # Consider one-hot encoding for high cardinality
                    one_hot = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), one_hot], axis=1)
                    self.logger.info(f"One-hot encoded high cardinality column: {col}")
        
        return df
    
    def scale_features(self, df, feature_columns):
        """
        Scale numerical features with configurable scaling method
        """
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        self.logger.info(f"Scaled features using {self.scaling_method} scaling: {feature_columns}")
        return df
    
    def process_data(self, df, target_column, categorical_columns=None):
        """
        Enhanced main data processing pipeline with better error handling
        """
        try:
            self.logger.info("Starting data processing pipeline")
            
            # Clean data
            df = self.clean_data(df)
            
            # Detect problem type
            problem_type = self.detect_problem_type(target_column, df)
            self.logger.info(f"Detected problem type: {problem_type}")
            
            # Handle categorical variables
            if categorical_columns is None:
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                categorical_columns = [col for col in categorical_columns if col != target_column]
            
            # Encode categorical variables
            if categorical_columns:
                df = self.encode_categorical(df, categorical_columns)
                self.logger.info(f"Encoded categorical columns: {categorical_columns}")
            
            # Scale numerical features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != target_column]
            if len(numeric_cols) > 0:
                df = self.scale_features(df, numeric_cols)
                self.logger.info(f"Scaled numeric columns: {numeric_cols}")
            
            # Verify all columns are numeric
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                self.logger.warning(f"Non-numeric columns remaining: {non_numeric_cols}")
                # Convert remaining non-numeric columns to numeric
                for col in non_numeric_cols:
                    if col != target_column:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.logger.info(f"Converted {col} to numeric")
            
            self.logger.info("Data processing pipeline completed successfully")
            return df, problem_type
            
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}", exc_info=True)
            raise

    def train_model(self, df, target_column, categorical_columns=None, model_type=None):
        """
        Train machine learning model based on problem type
        
        Args:
            df: Processed DataFrame
            target_column: Name of target column
            categorical_columns: List of categorical column names
            model_type: Type of model to train ('linear', 'knn', 'logistic', 'kmeans')
            
        Returns:
            Trained model and performance metrics
        """
        try:
            # Process data if not already processed
            if categorical_columns:
                df = self.encode_categorical(df, categorical_columns)
            
            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect problem type if not specified
            problem_type = self.detect_problem_type(target_column, df)
            
            # Train appropriate model
            if model_type == 'kmeans':
                model, metrics = self.ml_models.perform_clustering(X)
            elif problem_type == 'classification':
                model, metrics = self.ml_models.train_classification(X, y, model_type=model_type or 'logistic')
            else:  # regression
                model, metrics = self.ml_models.train_regression(X, y, model_type=model_type or 'linear')
            
            self.logger.info(f"Trained {model_type or problem_type} model successfully")
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training model: {str(e)}", exc_info=True)
            raise

    def predict(self, df, model_name):
        """
        Make predictions using a trained model
        
        Args:
            df: DataFrame to make predictions on
            model_name: Name of the trained model to use
            
        Returns:
            Predictions
        """
        try:
            return self.ml_models.predict(df, model_name)
        except Exception as e:
            self.logger.error(f"Error in making predictions: {str(e)}", exc_info=True)
            raise

    def save_model(self, model_name, path):
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model to save
            path: Path to save the model
        """
        try:
            self.ml_models.save_model(model_name, path)
        except Exception as e:
            self.logger.error(f"Error in saving model: {str(e)}", exc_info=True)
            raise

    def load_model(self, model_name, path):
        """
        Load a saved model from disk
        
        Args:
            model_name: Name to give the loaded model
            path: Path to the saved model
        """
        try:
            self.ml_models.load_model(model_name, path)
        except Exception as e:
            self.logger.error(f"Error in loading model: {str(e)}", exc_info=True)
            raise

def parse_prompt(prompt, df_columns):
    """
    Rule-based function to infer problem type, target column, and feature columns from a natural language prompt and dataframe columns.
    Args:
        prompt (str): User's natural language task description
        df_columns (list): List of dataframe column names
    Returns:
        dict: {'problem_type': ..., 'target': ..., 'features': ...}
    """
    prompt_lower = prompt.lower()
    columns_lower = [col.lower() for col in df_columns]
    
    # 1. Guess problem type
    if any(word in prompt_lower for word in ['classify', 'classification', 'yes/no', 'predict if', 'is it', 'will it', 'does it']):
        problem_type = 'classification'
    elif any(word in prompt_lower for word in ['regress', 'regression', 'predict amount', 'how much', 'price', 'income', 'predict value', 'estimate']):
        problem_type = 'regression'
    elif any(word in prompt_lower for word in ['predict', 'estimate']):
        # fallback: if target is numeric, regression; else classification
        problem_type = 'auto'
    else:
        problem_type = 'auto'
    
    # 2. Guess target column
    # Look for column names in prompt
    target = None
    for col, col_l in zip(df_columns, columns_lower):
        if col_l in prompt_lower:
            target = col
            break
    # Fuzzy match if not found
    if not target:
        words = re.findall(r'\w+', prompt_lower)
        matches = []
        for word in words:
            close = get_close_matches(word, columns_lower, n=1, cutoff=0.8)
            if close:
                idx = columns_lower.index(close[0])
                matches.append(df_columns[idx])
        if matches:
            target = matches[0]
    # Fallback: look for common target words
    if not target:
        for common in ['target', 'label', 'output', 'y', 'result', 'price', 'income', 'amount', 'score']:
            if common in columns_lower:
                target = df_columns[columns_lower.index(common)]
                break
    # If still not found, set to None
    
    # 3. Guess feature columns (all except target, drop id columns)
    features = [col for col in df_columns if col != target and not col.lower().startswith('id')]
    
    # 4. If problem_type is 'auto', guess from target column type
    if problem_type == 'auto' and target:
        # This function doesn't have access to the dataframe, so leave as 'auto' for now
        pass
    
    return {
        'problem_type': problem_type,
        'target': target,
        'features': features
    } 

def parse_prompt_with_openai(prompt, df_columns):
    """
    Enhanced OpenAI API integration with better error handling
    """
    import openai
    import json
    
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Please set it in the .env file")
    
    openai.api_key = OPENAI_API_KEY
    
    system_prompt = (
        f"Given the following dataframe columns: {df_columns}, "
        f"and the user task: '{prompt}', "
        "identify the problem type (classification/regression), the target column, and the relevant features. "
        "Return a JSON object with keys: problem_type, target, features (features as a list)."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}]
        )
        
        content = response['choices'][0]['message']['content']
        # Find the first JSON object in the content
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
        result = json.loads(json_str)
        return result
        
    except openai.error.AuthenticationError:
        raise ValueError("Invalid OpenAI API key")
    except openai.error.RateLimitError:
        raise Exception("OpenAI API rate limit exceeded")
    except Exception as e:
        raise Exception(f"Error in OpenAI API call: {str(e)}")

# Removed: result = parse_prompt_with_openai(prompt, df.columns.tolist()) 