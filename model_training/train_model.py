import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

def load_data(csv_path):
    return pd.read_csv(csv_path)

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
    return results

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == '__main__':
    # Get the absolute path to features.csv
    features_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_training/features.csv')
    print(f"Loading data from: {features_path}")
    
    # Load data
    data = load_data(features_path)
    # Keep only rows where Script_ID starts with 'JS_'
    data = data[data['Script_ID'].astype(str).str.startswith('JS_')]
    # Convert all boolean-like values to 1/0 for all columns
    for col in data.columns:
        if data[col].dtype == object and data[col].isin(['True', 'False', 'TRUE', 'FALSE', 1, 0]).any():
            data[col] = data[col].map({'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0, 1: 1, 0: 0})
        elif data[col].dtype == bool:
            data[col] = data[col].astype(int)
    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='ignore')

    # Debug: print columns with NaN values and their unique values
    nan_cols = data.columns[data.isna().any()]
    if len(nan_cols) > 0:
        print('Columns with NaN values after conversion:')
        for col in nan_cols:
            print(f"{col}: {data[col].unique()}")

    X = data.drop(['Script_ID', 'Label'], axis=1)
    y = data['Label'].map({'Benign': 0, 'Malicious': 1})
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    for model_name, metrics in evaluation_results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            if metric != 'Confusion Matrix':
                print(f"{metric}: {value}")
        print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
    
    # Save the best model (e.g., Random Forest)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rf_model.pkl')
    save_model(trained_models['Random Forest'], model_path)
    print(f"\nModel saved to: {model_path}") 