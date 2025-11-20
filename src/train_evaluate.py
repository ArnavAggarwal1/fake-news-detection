import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_models(X, y, trained_models=None):
    """
    Train multiple ML models with hyperparameter tuning and return them.
    Supports resuming from partial training.
    """
    if trained_models is None:
        trained_models = {}

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced'),
        'SVM': LinearSVC(class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Naive Bayes': {'alpha': [0.1, 0.5, 1.0]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'SVM': {'C': [0.1, 1, 10]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.2]}
    }

    for name, model in tqdm(models.items(), desc="Training Models"):
        if name in trained_models:
            print(f"Skipping {name}, already trained.")
            continue
        print(f"Training {name}...")
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X, y)
            trained_models[name] = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")
        else:
            model.fit(X, y)
            trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate models and return metrics, including cross-validation scores if training data is provided.
    """
    results = {}
    for name, model in tqdm(models.items(), desc="Evaluating Models"):
        y_pred = model.predict(X_test)
        y_prob = model.decision_function(X_test) if hasattr(model, 'decision_function') else (model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        # Cross-validation scores if training data is provided
        cv_scores = None
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_f1_mean': cv_scores.mean() if cv_scores is not None else None,
            'cv_f1_std': cv_scores.std() if cv_scores is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    return results

def plot_metrics(results):
    """
    Plot comparison of metrics.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    values = [[results[model][m] for m in metrics] for model in models]

    x = range(len(metrics))
    for i, model in enumerate(models):
        plt.bar([p + i*0.2 for p in x], values[i], width=0.2, label=model)

    plt.xticks([p + 0.3 for p in x], metrics)
    plt.legend()
    plt.title('Model Comparison')
    plt.show()

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for models.
    """
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    # plt.show()  # Removed to avoid display issues
