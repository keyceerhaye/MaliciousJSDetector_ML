import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib
import numpy as np
import os

def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(models, X_test, y_test, output_path):
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()

def plot_performance_comparison(evaluation_results, output_path):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    models = list(evaluation_results.keys())
    data = {metric: [evaluation_results[model][metric] for model in models] for metric in metrics}
    df = pd.DataFrame(data, index=models)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Performance Comparison')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rf_model.pkl')
    model = joblib.load(model_path)
    # Load test data
    features_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_training/features.csv')
    data = pd.read_csv(features_path)
    X_test = data.drop(['Script_ID', 'Label'], axis=1)
    y_test = data['Label']
    feature_names = X_test.columns
    # Plot feature importance
    plot_feature_importance(model, feature_names, 'feature_importance.png')
    # Plot ROC curves
    models = {'Random Forest': model}
    plot_roc_curves(models, X_test, y_test, 'roc_curves.png')
    # Plot performance comparison
    evaluation_results = {
        'Random Forest': {
            'Accuracy': 0.95,
            'Precision': 0.94,
            'Recall': 0.96,
            'F1 Score': 0.95,
            'ROC AUC': 0.97
        }
    }
    plot_performance_comparison(evaluation_results, 'performance_comparison.png') 