import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report  # <--- ADD THIS
)

# --- Configuration for your CSV ---
YOUR_CSV_FILE_PATH = 'Data.csv' # IMPORTANT: Change to 'scaled_features_with_label.csv' if that's your intended file
TARGET_COLUMN_NAME = 'label'
RANDOM_STATE = 42  # Defined globally for reproducibility

def load_data(csv_path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'. Please check the path and filename.")
        return None
    except Exception as e:
        print(f"Error loading data from '{csv_path}': {e}")
        return None

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = None):
    """Splits data into training and testing sets with stratification."""
    stratify_option = y if y.nunique() > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_option)

def train_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = None):
    """Initializes and trains multiple classifier models."""
    models = {
        'Random Forest': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        'SVM': SVC(random_state=random_state, probability=True, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000, solver='liblinear')
    }
    trained_models = {}
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} training complete.")
        except Exception as e:
            print(f"Error training {name}: {e}")
            trained_models[name] = None # Mark as None if training failed
    return trained_models

def plot_confusion_matrix_heatmap(y_true: pd.Series, y_pred: np.ndarray, class_names: list, model_name: str, output_dir: str = '.'):
    """
    Calculates, plots as a heatmap, and saves the confusion matrix for a specific model.
    """
    cm = confusion_matrix(y_true, y_pred)
    str_class_names = [str(cn) for cn in class_names]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=str_class_names, yticklabels=str_class_names,
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot for {model_name} saved to '{output_path}'.")
    plt.close()

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series, class_names: list, results_output_dir: str = 'model_results'):
    """
    Evaluates models, prints metrics, and saves confusion matrix plots.
    """
    results = {}
    print("\nEvaluating models...")

    if not os.path.exists(results_output_dir):
        os.makedirs(results_output_dir)

    for name, model in models.items():
        if model is None: # Skip if model training failed
            print(f"\n--- Skipping evaluation for {name} (training failed) ---")
            results[name] = {metric: np.nan for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}
            continue

        print(f"\n--- Evaluating {name} ---")
        y_pred = model.predict(X_test)
        
        roc_auc = np.nan
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC for {name}: {e}")
        else:
            print(f"Warning: {name} does not have predict_proba method. ROC AUC cannot be calculated.")

        current_metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC AUC': roc_auc
        }
        results[name] = current_metrics
        
        for metric, value in current_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        plot_confusion_matrix_heatmap(y_test, y_pred, class_names, name, output_dir=results_output_dir)
        
        print("\nDetailed Classification Report:")
        target_names_str = [str(cn) for cn in class_names]
        report = classification_report(y_test, y_pred, target_names=target_names_str, digits=4, zero_division=0)
        print(report)
        
    return results

def plot_model_comparison_metrics(evaluation_results: dict, output_dir: str = 'model_results'):
    """
    Creates and saves a grouped bar plot comparing key metrics across models.
    """
    # Filter out models that failed training (where all metrics are NaN)
    valid_results = {name: metrics for name, metrics in evaluation_results.items() if not all(pd.isna(v) for v in metrics.values())}
    if not valid_results:
        print("No valid model results to plot for comparison.")
        return

    metrics_df = pd.DataFrame(valid_results).T
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    plot_df = metrics_df[metrics_to_plot].copy()
    
    plot_df_melted = plot_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    plot_df_melted.rename(columns={'index': 'Model'}, inplace=True)

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Metric', y='Score', hue='Model', data=plot_df_melted, palette='muted')
    plt.title('Model Performance Comparison', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.05) 
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', fontsize=12, title_fontsize=13, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'model_comparison_metrics.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison plot saved to '{output_path}'.")
    plt.close()

def save_model(model, filename: str, model_dir: str = 'saved_models'):
    """Saves the model to a specified directory."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_path = os.path.join(model_dir, filename)
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")

if __name__ == '__main__':
    print("Starting Machine Learning Pipeline...")
    
    data = load_data(YOUR_CSV_FILE_PATH)
    if data is None:
        exit() # Exit if data loading failed
    
    # Ensure all feature columns are numeric 
    potential_feature_cols = [col for col in data.columns if col != TARGET_COLUMN_NAME]
    for col in potential_feature_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce') 
    
    # Drop rows with NaNs that might have been introduced by pd.to_numeric if any non-numeric data existed in feature columns
    # This should happen BEFORE splitting X and y
    original_rows = len(data)
    data.dropna(subset=potential_feature_cols, inplace=True)
    if len(data) < original_rows:
        print(f"Dropped {original_rows - len(data)} rows due to NaNs in feature columns after numeric conversion.")

    if data.empty:
        print("Error: Data is empty after attempting to convert features to numeric and dropping NaNs. Please check your CSV file.")
        exit()

    # Define features (X) and target (y)
    if TARGET_COLUMN_NAME not in data.columns:
        print(f"Error: Target column '{TARGET_COLUMN_NAME}' not found in the CSV file.")
        exit()
    
    X = data.drop(TARGET_COLUMN_NAME, axis=1)
    y = data[TARGET_COLUMN_NAME]

    # Final check for NaNs that might have been missed or in target column if it wasn't numeric initially
    if X.isna().any().any():
        print("\nCritical Warning: NaN values still present in features (X) before splitting. This will cause errors.")
        print(X.isna().sum()[X.isna().sum() > 0])
        # Aggressively drop rows with any NaN in X or handle by imputation
        X.dropna(inplace=True)
        y = y.loc[X.index] # Align y
        if X.empty:
             print("Error: Features (X) became empty after final NaN drop. Review data.")
             exit()
    if y.isna().any():
        print("\nCritical Warning: NaN values present in target (y) before splitting.")
        print(y.isna().sum())
        valid_indices = y.dropna().index
        y = y.loc[valid_indices]
        X = X.loc[valid_indices]
        if y.empty:
            print("Error: Target (y) became empty after final NaN drop. Review data.")
            exit()

    try:
        y = y.astype(int)
    except ValueError as e:
        print(f"Error converting target column '{TARGET_COLUMN_NAME}' to integer: {e}")
        print(f"Unique values in target column before conversion attempt: {data[TARGET_COLUMN_NAME].unique()}")
        exit()


    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    if y.empty:
        print("Target variable y is empty. Cannot proceed.")
        exit()
    print(f"Target (y) unique values: {y.unique()}")
    print(f"Target (y) class distribution:\n{y.value_counts(normalize=True)}")


    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_STATE) 
    
    trained_models = train_models(X_train, y_train, random_state=RANDOM_STATE) 
    
    if not y_test.empty: # Ensure y_test is not empty before getting unique values
        class_names_for_eval = sorted(y_test.unique().tolist())
    else:
        print("Warning: y_test is empty. Cannot determine class names for evaluation plots.")
        class_names_for_eval = [] # Default to empty list if y_test is empty

    results_dir = "model_evaluation_results"
    evaluation_results = evaluate_models(trained_models, X_test, y_test, class_names_for_eval, results_output_dir=results_dir)
            
    if evaluation_results: # Only plot if there are results
        plot_model_comparison_metrics(evaluation_results, output_dir=results_dir)
    
    # Save the best model (e.g., based on F1 Score or ROC AUC)
    # For now, just saving Random Forest as an example, if it was trained.
    best_model_name = 'Random Forest' 
    if best_model_name in trained_models and trained_models[best_model_name] is not None:
        model_save_dir = "saved_trained_models"
        model_filename = f'{best_model_name.lower().replace(" ", "_")}_model.pkl'
        save_model(trained_models[best_model_name], model_filename, model_dir=model_save_dir)
    else:
        print(f"\nCould not save {best_model_name} model as it was not trained successfully or is not in trained_models.")
    
    print("\nMachine Learning Pipeline completed.")