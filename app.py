import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, boxcox, chi2_contingency
from data_processor import DataProcessor, parse_prompt
from model_trainer import ModelTrainer
from ml_models import MLModels
import logging
import os
import joblib
import shap
import lime
from datetime import datetime
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_eda(df):
    """Perform comprehensive EDA on the dataset"""
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # 1. Basic Information
    st.markdown("### 📌 Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
    with col2:
        st.write("Columns:", df.columns.tolist())
    
    # 2. Data Types and Missing Values
    st.markdown("### 📌 Data Types and Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Types:")
        st.write(df.dtypes)
    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())
    
    # 3. Feature Classification
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    
    st.markdown("### 📌 Feature Classification")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Numerical Features:", numerical_features)
    with col2:
        st.write("Categorical Features:", categorical_features)
    
    # 4. Numerical Features Analysis
    if numerical_features:
        st.markdown("### 📌 Numerical Features Analysis")
        
        # Descriptive Statistics
        st.write("Descriptive Statistics:")
        st.write(df[numerical_features].describe())
        
        # Distribution Plots
        st.write("Distribution Plots:")
        for col in numerical_features:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)
            
            # Normality Test
            stat, p = shapiro(df[col].sample(min(500, len(df))))
            st.write(f"{col} - Shapiro Test: W={stat:.4f}, p={p:.4f}")
            
            # Skewness and Kurtosis
            skew = df[col].skew()
            kurt = df[col].kurtosis()
            st.write(f"{col} - Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")
            
            # Box-Cox Transformation (if applicable)
            if (df[col] > 0).all():
                transformed, lam = boxcox(df[col])
                st.write(f"{col} - Box-Cox λ = {lam:.4f}")
    
    # 5. Categorical Features Analysis
    if categorical_features:
        st.markdown("### 📌 Categorical Features Analysis")
        
        for col in categorical_features:
            st.write(f"\n{col} Frequency Distribution:")
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Frequency Distribution of {col}")
            st.plotly_chart(fig)
    
    # 6. Correlation Analysis
    if len(numerical_features) > 1:
        st.markdown("### 📌 Correlation Analysis")
        corr = df[numerical_features].corr()
        fig = px.imshow(corr, 
                       title="Correlation Heatmap",
                       labels=dict(color="Correlation"))
        st.plotly_chart(fig)
    
    # 7. Pairplot for selected features
    if len(numerical_features) > 1:
        st.markdown("### 📌 Feature Relationships")
        selected_features = st.multiselect(
            "Select features for pairplot",
            numerical_features,
            default=numerical_features[:min(5, len(numerical_features))]
        )
        if selected_features:
            fig = px.scatter_matrix(df, dimensions=selected_features)
            st.plotly_chart(fig)
    
    # 8. Category-wise Analysis
    if categorical_features and numerical_features:
        st.markdown("### 📌 Category-wise Analysis")
        cat_col = st.selectbox("Select categorical feature", categorical_features)
        num_col = st.selectbox("Select numerical feature", numerical_features)
        
        fig = px.box(df, x=cat_col, y=num_col, 
                    title=f"{num_col} by {cat_col}")
        st.plotly_chart(fig)
        
        fig = px.violin(df, x=cat_col, y=num_col,
                       title=f"Distribution of {num_col} by {cat_col}")
        st.plotly_chart(fig)
    
    # 9. Time Series Analysis (if applicable)
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_columns:
        st.markdown("### 📌 Time Series Analysis")
        date_col = st.selectbox("Select date column", date_columns)
        target_col = st.selectbox("Select target column", numerical_features)
        
        if st.button("Perform Time Series Analysis"):
            ml_models = MLModels()
            model, metrics = ml_models.train_time_series(df, date_col, target_col)
            
            # Plot time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col], y=df[target_col],
                                   mode='lines', name='Actual'))
            
            # Add forecast
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                   mode='lines', name='Forecast'))
            
            fig.update_layout(title=f"Time Series Analysis: {target_col}",
                            xaxis_title=date_col,
                            yaxis_title=target_col)
            st.plotly_chart(fig)
            
            # Show metrics
            st.write("Time Series Metrics:", metrics)

def data_science_assistant():
    st.title("AI-Powered Data Science Assistant")
    st.write("Upload your dataset and describe your business task to get started!")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            
            # Task description
            task_description = st.text_area(
                "Describe your business task (e.g., 'predict discount on hatchbacks')",
                height=100
            )
            
            if task_description:
                # --- Prompt Parsing ---
                st.info("🔍 Parsing your prompt and inferring task...")
                parsed = parse_prompt(task_description, df.columns.tolist())
                
                # Show what the AI inferred
                st.write("**AI Inferred:**")
                st.write(parsed)
                
                # Allow user to override if needed
                st.markdown("---")
                st.write("### Confirm or Edit the AI's Selections:")
                problem_type = st.selectbox(
                    "Problem Type",
                    ["classification", "regression"],
                    index=["classification", "regression"].index(parsed['problem_type']) if parsed['problem_type'] in ["classification", "regression"] else 0
                )
                target_column = st.selectbox(
                    "Target Column",
                    df.columns,
                    index=df.columns.get_loc(parsed['target']) if parsed['target'] in df.columns else 0
                )
                feature_columns = st.multiselect(
                    "Feature Columns",
                    [col for col in df.columns if col != target_column],
                    default=[col for col in parsed['features'] if col in df.columns and col != target_column]
                )
                
                # --- EDA ---
                st.markdown("---")
                st.header("Exploratory Data Analysis (EDA)")
                perform_eda(df)
                
                # --- Data Processing & Model Training ---
                if target_column and len(feature_columns) > 0:
                    data_processor = DataProcessor()
                    ml_models = MLModels()
                    categorical_columns = df[feature_columns].select_dtypes(include=['object']).columns.tolist()
                    
                    with st.spinner("Processing data and training model..."):
                        processed_df, _ = data_processor.process_data(
                            df,
                            target_column,
                            categorical_columns=categorical_columns
                        )
                        
                        # Model selection
                        st.markdown("### Model Selection")
                        if problem_type == "classification":
                            model_type = st.selectbox(
                                "Select Model Type",
                                ["logistic", "knn", "rf", "gb", "svc", "xgboost", "lightgbm", "catboost"]
                            )
                        else:  # regression
                            model_type = st.selectbox(
                                "Select Model Type",
                                ["linear", "ridge", "lasso", "knn", "rf", "gb", "svr", "xgboost", "lightgbm", "catboost"]
                            )
                        
                        X = processed_df[feature_columns]
                        y = processed_df[target_column]
                        
                        if problem_type == "classification":
                            model, metrics = ml_models.train_classification(X, y, model_type)
                        else:
                            model, metrics = ml_models.train_regression(X, y, model_type)
                        
                        st.success("Model trained successfully!")
                        
                        # Model Performance
                        st.subheader("Model Performance")
                        st.write(metrics)
                        
                        # Feature Importance
                        st.subheader("Feature Importance")
                        if model_type in ['rf', 'gb', 'xgboost', 'lightgbm', 'catboost']:
                            importance_df = pd.DataFrame({
                                'feature': feature_columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x='feature',
                                y='importance',
                                title='Feature Importance'
                            )
                            st.plotly_chart(fig)
                        
                        # Model Explanation
                        st.subheader("Model Explanation")
                        if st.checkbox("Show Model Explanation"):
                            explanation = ml_models.get_model_explanation(
                                f"{model_type}_{problem_type}",
                                X,
                                instance_index=0
                            )
                            
                            if 'shap_values' in explanation:
                                st.write("SHAP Values:")
                                shap_values = explanation['shap_values']
                                fig = shap.summary_plot(
                                    shap_values,
                                    X,
                                    plot_type="bar",
                                    show=False
                                )
                                st.pyplot(fig)
                            
                            if 'lime_explanation' in explanation:
                                st.write("LIME Explanation:")
                                lime_exp = explanation['lime_explanation']
                                st.write(lime_exp)
                        
                        # Download Processed Data
                        st.subheader("Download Processed Data")
                        processed_csv = processed_df.to_csv(index=False)
                        st.download_button(
                            "Download Processed CSV",
                            processed_csv,
                            "processed_data.csv",
                            "text/csv"
                        )
                else:
                    st.warning("Please confirm both target and feature columns to proceed.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main app: {str(e)}")

def main():
    # Set page config once at the start
    st.set_page_config(
        page_title="AI Data Science Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Only show Data Science Assistant
    data_science_assistant()

if __name__ == "__main__":
    main()