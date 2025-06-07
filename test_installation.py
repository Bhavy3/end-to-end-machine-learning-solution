import sys
import importlib
import logging

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'streamlit',
        'plotly',
        'seaborn',
        'matplotlib',
        'xgboost',
        'lightgbm',
        'catboost',
        'prophet',
        'shap',
        'lime',
        'optuna',
        'fastapi',
        'uvicorn'
    ]
    
    print("Testing package imports...")
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    if missing_packages:
        print("\nMissing packages:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install missing packages using:")
        print("pip install " + " ".join(missing_packages))
    else:
        print("\n✅ All required packages are installed!")

if __name__ == "__main__":
    test_imports() 