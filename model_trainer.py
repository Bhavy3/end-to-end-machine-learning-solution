import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import logging

class ModelTrainer:
    """
    Handles model selection, training, evaluation, feature importance, and saving/loading.
    Automatically adapts to classification or regression tasks.
    """
    def __init__(self):
        self.model = None
        self.problem_type = None
        self.logger = logging.getLogger(__name__)

    def select_model(self, problem_type):
        """
        Selects and returns a model based on problem type.
        Extendable for more models in the future.
        """
        if problem_type == 'classification':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self, X, y, problem_type, cv_folds=0):
        """
        Trains the model and returns performance metrics.
        Optionally performs cross-validation if cv_folds > 1.
        """
        self.problem_type = problem_type
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.model = self.select_model(problem_type)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if problem_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                if cv_folds and cv_folds > 1:
                    metrics['cv_accuracy'] = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy').mean()
            else:
                metrics = {
                    'rmse': mean_squared_error(y_test, y_pred, squared=False),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                if cv_folds and cv_folds > 1:
                    metrics['cv_rmse'] = -cross_val_score(self.model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error').mean()
            return metrics
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def get_feature_importance(self, feature_names):
        """
        Returns a DataFrame of feature importances.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            return feature_importance.sort_values('importance', ascending=False)
        else:
            raise ValueError("Model does not support feature importances.")

    def save_model(self, path):
        """
        Saves the trained model to the given path.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Loads a model from the given path.
        """
        self.model = joblib.load(path) 