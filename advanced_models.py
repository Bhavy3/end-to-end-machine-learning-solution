import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from prophet import Prophet
import optuna
from sklearn.metrics import mean_squared_error, accuracy_score
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    PROPHET_PARAMS, SHAP_SETTINGS, LIME_SETTINGS,
    OPTUNA_SETTINGS
)

class AdvancedModels:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.explainers = {}
        
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                     problem_type: str = 'regression') -> Tuple[Any, Dict[str, float]]:
        """Train XGBoost model with Optuna optimization"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            if problem_type == 'classification':
                params['objective'] = 'binary:logistic'
                model = xgb.XGBClassifier(**params)
            else:
                params['objective'] = 'reg:squarederror'
                model = xgb.XGBRegressor(**params)
                
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if problem_type == 'classification':
                return accuracy_score(y, y_pred)
            return -mean_squared_error(y, y_pred)
        
        study = optuna.create_study(direction='maximize' if problem_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=OPTUNA_SETTINGS['n_trials'])
        
        best_params = study.best_params
        if problem_type == 'classification':
            model = xgb.XGBClassifier(**best_params)
        else:
            model = xgb.XGBRegressor(**best_params)
            
        model.fit(X, y)
        self.models['xgboost'] = model
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        self.explainers['xgboost'] = explainer
        
        return model, {'best_params': best_params, 'best_value': study.best_value}
    
    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                      problem_type: str = 'regression') -> Tuple[Any, Dict[str, float]]:
        """Train LightGBM model with Optuna optimization"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            if problem_type == 'classification':
                params['objective'] = 'binary'
                model = lgb.LGBMClassifier(**params)
            else:
                params['objective'] = 'regression'
                model = lgb.LGBMRegressor(**params)
                
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if problem_type == 'classification':
                return accuracy_score(y, y_pred)
            return -mean_squared_error(y, y_pred)
        
        study = optuna.create_study(direction='maximize' if problem_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=OPTUNA_SETTINGS['n_trials'])
        
        best_params = study.best_params
        if problem_type == 'classification':
            model = lgb.LGBMClassifier(**best_params)
        else:
            model = lgb.LGBMRegressor(**best_params)
            
        model.fit(X, y)
        self.models['lightgbm'] = model
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        self.explainers['lightgbm'] = explainer
        
        return model, {'best_params': best_params, 'best_value': study.best_value}
    
    def train_catboost(self, X: pd.DataFrame, y: pd.Series,
                      problem_type: str = 'regression') -> Tuple[Any, Dict[str, float]]:
        """Train CatBoost model with Optuna optimization"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0)
            }
            
            if problem_type == 'classification':
                model = CatBoostClassifier(**params, verbose=False)
            else:
                model = CatBoostRegressor(**params, verbose=False)
                
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if problem_type == 'classification':
                return accuracy_score(y, y_pred)
            return -mean_squared_error(y, y_pred)
        
        study = optuna.create_study(direction='maximize' if problem_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=OPTUNA_SETTINGS['n_trials'])
        
        best_params = study.best_params
        if problem_type == 'classification':
            model = CatBoostClassifier(**best_params, verbose=False)
        else:
            model = CatBoostRegressor(**best_params, verbose=False)
            
        model.fit(X, y)
        self.models['catboost'] = model
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        self.explainers['catboost'] = explainer
        
        return model, {'best_params': best_params, 'best_value': study.best_value}
    
    def get_shap_values(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for model predictions"""
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model: {model_name}")
            
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X)
        return shap_values
    
    def get_lime_explanation(self, model_name: str, X: pd.DataFrame, 
                           instance_index: int) -> Dict[str, Any]:
        """Get LIME explanation for a specific instance"""
        if model_name not in self.models:
            raise ValueError(f"No model found: {model_name}")
            
        explainer = LimeTabularExplainer(
            X.values,
            feature_names=X.columns,
            class_names=['target'],
            mode='regression' if isinstance(self.models[model_name], 
                                         (xgb.XGBRegressor, lgb.LGBMRegressor, CatBoostRegressor)) 
                  else 'classification'
        )
        
        exp = explainer.explain_instance(
            X.iloc[instance_index].values,
            self.models[model_name].predict,
            num_features=LIME_SETTINGS['num_features']
        )
        
        return {
            'explanation': exp.as_list(),
            'prediction': exp.predicted_value
        }
    
    def train_prophet(self, df: pd.DataFrame, date_col: str, 
                     target_col: str) -> Tuple[Any, Dict[str, float]]:
        """Train Prophet model for time series forecasting"""
        prophet_df = df.rename(columns={date_col: 'ds', target_col: 'y'})
        model = Prophet(**PROPHET_PARAMS)
        model.fit(prophet_df)
        self.models['prophet'] = model
        
        # Calculate metrics
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(prophet_df['y'], 
                                             forecast['yhat'][:len(prophet_df)]))
        }
        
        return model, metrics