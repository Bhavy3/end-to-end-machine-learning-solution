import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any, Union, List
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from advanced_models import AdvancedModels
import shap
from lime.lime_tabular import LimeTabularExplainer

class MLModels:
    def __init__(self):
        self.models = {}
        self._setup_logging()
        self.best_params = {}
        self.scaler = StandardScaler()
        self.model_history = []
        self.advanced_models = AdvancedModels()
        self.explainers = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_history.append({
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics
        })
        self.logger.info(f"Model {model_name} performance at {timestamp}: {metrics}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot confusion matrix for classification models"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'plots/{model_name}_confusion_matrix.png')
        plt.close()
    
    def _plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/{model_name}_feature_importance.png')
            plt.close()
    
    def train_regression(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str = 'linear') -> Tuple[Any, Dict[str, float]]:
        """
        Train regression models with enhanced options
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'linear', 'ridge', 'lasso', 'knn', 'rf', 'gb', 'svr', 'xgboost', 'lightgbm', 'catboost'
            
        Returns:
            Trained model and performance metrics
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            if model_type in ['xgboost', 'lightgbm', 'catboost']:
                # Use advanced models
                if model_type == 'xgboost':
                    model, metrics = self.advanced_models.train_xgboost(X_train, y_train, 'regression')
                elif model_type == 'lightgbm':
                    model, metrics = self.advanced_models.train_lightgbm(X_train, y_train, 'regression')
                else:  # catboost
                    model, metrics = self.advanced_models.train_catboost(X_train, y_train, 'regression')
            else:
                # Use traditional models
                if model_type == 'linear':
                    model = LinearRegression()
                elif model_type == 'ridge':
                    model = Ridge()
                elif model_type == 'lasso':
                    model = Lasso()
                elif model_type == 'knn':
                    model = KNeighborsRegressor()
                elif model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == 'gb':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_type == 'svr':
                    model = SVR()
                else:
                    raise ValueError(f"Unsupported regression model type: {model_type}")
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'explained_variance': explained_variance_score(y_test, y_pred)
                }
            
            # Store model and create explainer
            model_key = f'{model_type}_regression'
            self.models[model_key] = model
            
            # Create SHAP explainer for supported models
            if model_type in ['rf', 'gb', 'xgboost', 'lightgbm', 'catboost']:
                self.explainers[model_key] = shap.TreeExplainer(model)
            elif model_type in ['linear', 'ridge', 'lasso']:
                self.explainers[model_key] = shap.LinearExplainer(model, X_train_scaled)
            
            self._log_model_performance(model_key, metrics)
            
            # Plot feature importance for supported models
            if model_type in ['rf', 'gb', 'xgboost', 'lightgbm', 'catboost']:
                self._plot_feature_importance(model, X.columns, model_key)
            elif model_type in ['linear', 'ridge', 'lasso']:
                # For linear models, use coefficients as feature importance
                importance = np.abs(model.coef_)
                self._plot_feature_importance(importance, X.columns, model_key)
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training regression model: {str(e)}")
            raise
    
    def train_classification(self, X: pd.DataFrame, y: pd.Series,
                           model_type: str = 'logistic') -> Tuple[Any, Dict[str, float]]:
        """
        Train classification models with enhanced options
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'logistic', 'knn', 'rf', 'gb', 'svc', 'xgboost', 'lightgbm', 'catboost'
            
        Returns:
            Trained model and performance metrics
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            if model_type in ['xgboost', 'lightgbm', 'catboost']:
                # Use advanced models
                if model_type == 'xgboost':
                    model, metrics = self.advanced_models.train_xgboost(X_train, y_train, 'classification')
                elif model_type == 'lightgbm':
                    model, metrics = self.advanced_models.train_lightgbm(X_train, y_train, 'classification')
                else:  # catboost
                    model, metrics = self.advanced_models.train_catboost(X_train, y_train, 'classification')
                
                y_pred = model.predict(X_test)
                metrics.update({
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                })
                
                # Add ROC AUC for binary classification
                if len(np.unique(y)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                # Use traditional models
                if model_type == 'logistic':
                    model = LogisticRegression(max_iter=1000)
                elif model_type == 'knn':
                    param_grid = {'n_neighbors': range(1, 21)}
                    model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
                elif model_type == 'rf':
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
                elif model_type == 'gb':
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    }
                    model = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
                elif model_type == 'svc':
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf']
                    }
                    model = GridSearchCV(SVC(probability=True), param_grid, cv=5)
                else:
                    raise ValueError(f"Unsupported classification model type: {model_type}")
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Add ROC AUC for binary classification
                if len(np.unique(y)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
            
            if hasattr(model, 'best_params_'):
                self.best_params[f'{model_type}_classification'] = model.best_params_
                model = model.best_estimator_
            
            self.models[f'{model_type}_classification'] = model
            self._log_model_performance(f'{model_type}_classification', metrics)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_test, y_pred, f'{model_type}_classification')
            
            # Plot feature importance for tree-based models
            if model_type in ['rf', 'gb', 'xgboost', 'lightgbm', 'catboost']:
                self._plot_feature_importance(model, X.columns, f'{model_type}_classification')
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training classification model: {str(e)}")
            raise
    
    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = None,
                          method: str = 'kmeans') -> Tuple[Any, Dict[str, float]]:
        """
        Perform clustering with multiple methods
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, will find optimal k)
            method: 'kmeans' or 'dbscan'
            
        Returns:
            Trained clustering model and performance metrics
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if method == 'kmeans':
                if n_clusters is None:
                    # Find optimal number of clusters using silhouette score
                    silhouette_scores = []
                    K = range(2, 11)
                    
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(X_scaled)
                        score = silhouette_score(X_scaled, kmeans.labels_)
                        silhouette_scores.append(score)
                    
                    n_clusters = K[np.argmax(silhouette_scores)]
                    self.logger.info(f"Optimal number of clusters: {n_clusters}")
                
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X_scaled)
                
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                model.fit(X_scaled)
                n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
            
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            metrics = {
                'silhouette_score': silhouette_score(X_scaled, model.labels_),
                'n_clusters': n_clusters
            }
            
            if method == 'kmeans':
                metrics['inertia'] = model.inertia_
            
            self.models[f'{method}_clustering'] = model
            self._log_model_performance(f'{method}_clustering', metrics)
            
            # Plot cluster visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap='viridis')
            plt.title(f'Cluster Visualization - {method}')
            plt.savefig(f'plots/{method}_clusters.png')
            plt.close()
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in performing clustering: {str(e)}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      model_name: str, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a trained model
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of the trained model
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.models[model_name]
        X_scaled = self.scaler.fit_transform(X)
        
        if 'classification' in model_name:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['neg_mean_squared_error', 'r2', 'explained_variance']
        
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        self.logger.info(f"Cross-validation results for {model_name}: {cv_results}")
        return cv_results
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            X: Feature matrix
            model_name: Name of the trained model to use
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Get probability estimates for classification models
        
        Args:
            X: Feature matrix
            model_name: Name of the trained model to use
            
        Returns:
            Probability estimates
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        if not hasattr(self.models[model_name], 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support probability estimates")
        
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict_proba(X_scaled)
    
    def save_model(self, model_name: str, path: str):
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model to save
            path: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models[model_name], path)
        self.logger.info(f"Saved model {model_name} to {path}")
    
    def load_model(self, model_name: str, path: str):
        """
        Load a saved model from disk
        
        Args:
            model_name: Name to give the loaded model
            path: Path to the saved model
        """
        self.models[model_name] = joblib.load(path)
        self.logger.info(f"Loaded model {model_name} from {path}")
    
    def get_model_history(self) -> pd.DataFrame:
        """
        Get the history of model training and performance
        
        Returns:
            DataFrame containing model training history
        """
        return pd.DataFrame(self.model_history)
    
    def get_model_explanation(self, model_name: str, X: pd.DataFrame, 
                            instance_index: int = 0) -> Dict[str, Any]:
        """
        Get model explanation using SHAP and LIME
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"No model found: {model_name}")
            
            explanation = {}
            
            # Get SHAP values if explainer exists
            if model_name in self.explainers:
                explainer = self.explainers[model_name]
                shap_values = explainer.shap_values(X)
                explanation['shap_values'] = shap_values
            
            # Get LIME explanation
            if self.models[model_name].__class__.__name__ in ['LinearRegression', 'Ridge', 'Lasso']:
                mode = 'regression'
            else:
                mode = 'classification'
            
            explainer = LimeTabularExplainer(
                X.values,
                feature_names=X.columns,
                class_names=['target'],
                mode=mode
            )
            
            exp = explainer.explain_instance(
                X.iloc[instance_index].values,
                self.models[model_name].predict,
                num_features=10
            )
            
            explanation['lime_explanation'] = exp.as_list()
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error in getting model explanation: {str(e)}")
            raise
    
    def train_time_series(self, df: pd.DataFrame, date_col: str, 
                         target_col: str) -> Tuple[Any, Dict[str, float]]:
        """
        Train time series model using Prophet
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            target_col: Name of target column
            
        Returns:
            Trained model and performance metrics
        """
        try:
            model, metrics = self.advanced_models.train_prophet(df, date_col, target_col)
            self.models['prophet'] = model
            self._log_model_performance('prophet', metrics)
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training time series model: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from data_processor import DataProcessor
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    # Sample data
    your_dataframe = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'your_target': np.random.randint(0, 2, 100),
        'cat_col1': np.random.choice(['A', 'B', 'C'], 100),
        'cat_col2': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # Initialize processor
    processor = DataProcessor()
    
    # Process data
    df_processed, problem_type = processor.process_data(
        df=your_dataframe,
        target_column='your_target',
        categorical_columns=['cat_col1', 'cat_col2']
    )
    
    # Initialize ML models
    ml = MLModels()
    
    # Train different models
    X = df_processed.drop('your_target', axis=1)
    y = df_processed['your_target']
    
    # Train regression models
    model, metrics = ml.train_regression(X, y, model_type='rf')
    print("Random Forest Regression Metrics:", metrics)
    
    model, metrics = ml.train_regression(X, y, model_type='gb')
    print("Gradient Boosting Regression Metrics:", metrics)
    
    # Train classification models
    model, metrics = ml.train_classification(X, y, model_type='rf')
    print("Random Forest Classification Metrics:", metrics)
    
    # Clustering
    model, metrics = ml.perform_clustering(X, method='kmeans')
    print("K-means Clustering Metrics:", metrics)
    
    # Cross-validation
    cv_results = ml.cross_validate(X, y, 'rf_classification', cv=5)
    print("Cross-validation Results:", cv_results)
    
    # Get model history
    history = ml.get_model_history()
    print("Model History:", history) 