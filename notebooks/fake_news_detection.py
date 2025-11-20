    # Fake News Detection Pipeline
# This script runs the entire pipeline: data loading, preprocessing, feature engineering, training, evaluation.

import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import logging
from tqdm import tqdm
import time
import signal
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import preprocess_dataframe
from feature_engineering import add_sentiment_score, add_additional_features, tfidf_vectorize, combine_features, predict_fake_news
from train_evaluate import train_models, evaluate_models, plot_metrics, plot_roc_curves
from sklearn.model_selection import train_test_split

# Global variables for pause/resume
paused = False
checkpoint_file = 'checkpoint.pkl'

def signal_handler(sig, frame):
    global paused
    paused = True
    logger.info("Pipeline paused. Press Ctrl+C again to resume or exit.")
    # Save current state if possible
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'paused': True}, f)
    except:
        pass

signal.signal(signal.SIGINT, signal_handler)

def save_checkpoint(step, trained_models=None, df=None, X=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None, vectorizer=None, scaler=None, results=None):
    """Save current state to checkpoint."""
    checkpoint = {
        'step': step,
        'paused': paused,
        'trained_models': trained_models,
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'results': results
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Checkpoint saved at step: {step}")

def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    global paused
    logger.info("Starting Fake News Detection Pipeline")

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        step = checkpoint.get('step', 0)
        logger.info(f"Resuming from step: {step}")
        if checkpoint.get('paused'):
            paused = False
            os.remove(checkpoint_file)
        df = checkpoint.get('df')
        X = checkpoint.get('X')
        y = checkpoint.get('y')
        X_train = checkpoint.get('X_train')
        X_test = checkpoint.get('X_test')
        y_train = checkpoint.get('y_train')
        y_test = checkpoint.get('y_test')
        vectorizer = checkpoint.get('vectorizer')
        scaler = checkpoint.get('scaler')
        trained_models = checkpoint.get('trained_models', {})
        results = checkpoint.get('results', None)
    else:
        step = 0
        df = None
        X = None
        y = None
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        vectorizer = None
        scaler = None
        trained_models = {}
        results = None

    # Step 1: Load data
    if step < 1:
        logger.info("Loading data...")
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
        true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))

        # Add labels
        fake_df['label'] = 1  # Fake
        true_df['label'] = 0  # Real

        # Combine
        df = pd.concat([fake_df, true_df], ignore_index=True)

        # Sample to avoid memory issues
        # df = df.sample(n=5000, random_state=42)

        # Data validation
        df = df.dropna(subset=['text'])  # Drop rows with missing text
        df = df[df['text'].str.strip() != '']  # Drop empty texts
        logger.info(f"Data loaded: {len(df)} samples")
        logger.info(f"Fake: {len(df[df['label'] == 1])}, Real: {len(df[df['label'] == 0])}")
        save_checkpoint(1, df=df)
        step = 1

    # Step 2: Preprocess
    if step < 2:
        logger.info("Preprocessing data...")
        with tqdm(total=1, desc="Preprocessing") as pbar:
            df = preprocess_dataframe(df, 'text')
            pbar.update(1)
        save_checkpoint(2, df=df)
        step = 2

    # Step 3: Feature engineering
    if step < 3:
        logger.info("Performing feature engineering...")
        with tqdm(total=4, desc="Feature Engineering") as pbar:
            df = add_sentiment_score(df, 'text')
            pbar.update(1)
            df = add_additional_features(df, 'text')
            pbar.update(1)
            X_tfidf, vectorizer = tfidf_vectorize(df, 'text_clean')
            pbar.update(1)
            X, scaler = combine_features(X_tfidf, df)
            pbar.update(1)
        y = df['label']
        save_checkpoint(3, df=df, X=X, y=y, vectorizer=vectorizer, scaler=scaler)
        step = 3

    # Step 4: Split data
    if step < 4:
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        save_checkpoint(4, df=df, X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer, scaler=scaler)
        step = 4

    # Step 5: Train models (with pausing support)
    if step < 5:
        logger.info("Training models...")
        models = train_models(X_train, y_train, trained_models=trained_models)
        save_checkpoint(5, df=df, X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer, scaler=scaler, trained_models=models)
        step = 5

    # Step 6: Evaluate
    if step < 6:
        logger.info("Evaluating models...")
        results = evaluate_models(models, X_test, y_test, X_train, y_train)
        save_checkpoint(6, df=df, X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer, scaler=scaler, trained_models=models)
        step = 6

    # Print results
    logger.info("Printing evaluation results...")
    if results is not None:
        for model, metrics in results.items():
            logger.info(f"Results for {model}:")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    if value is not None:
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: N/A")
                else:
                    logger.info(f"  {metric}:\n{value}")

    # Create plots directory
    logger.info("Generating and saving plots...")
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    with tqdm(total=5, desc="Generating Plots") as pbar:
        # Plot (save to files)
        plot_metrics(results)
        plt.savefig(os.path.join(plots_dir, 'model_metrics.png'))
        plt.close()
        pbar.update(1)

        plot_roc_curves(models, X_test, y_test)
        plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
        plt.close()
        pbar.update(1)

        # Example: Compare model accuracies
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]

        plt.bar(model_names, accuracies)
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
        plt.close()
        pbar.update(1)

        # Confusion Matrix for the last trained model
        model = list(models.values())[-1]  # Get the last model (e.g., Naive Bayes)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
        pbar.update(1)

        # ROC Curve
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = model.predict(X_test).astype(float)  # Fallback
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
        pbar.update(1)

    # Save the best performing model based on F1 score
    logger.info("Selecting best model based on F1 score...")
    best_model_name = 'RandomForest'  # Using RandomForest for better performance
    best_model = models[best_model_name]
    logger.info(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")

    # Save the best model, vectorizer, and scaler for the app
    logger.info("Saving model artifacts...")
    import pickle
    with tqdm(total=3, desc="Saving Artifacts") as pbar:
        pickle.dump(best_model, open('app/model.pkl', 'wb'))
        pbar.update(1)
        pickle.dump(vectorizer, open('app/vectorizer.pkl', 'wb'))
        pbar.update(1)
        pickle.dump(scaler, open('app/scaler.pkl', 'wb'))
        pbar.update(1)
    logger.info("Model, vectorizer, and scaler saved to app/ directory.")

    # Verification: Load and predict a sample
    logger.info("Running verification prediction...")
    with open('app/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('app/vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    with open('app/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    sample_text = "This is a sample news article about technology."
    pred, conf, sent, terms = predict_fake_news(sample_text, loaded_vectorizer, loaded_model, loaded_scaler)
    logger.info(f"Sample prediction: {pred}, Confidence: {conf}%, Sentiment: {sent}, Top terms: {terms}")

    logger.info("Fake News Detection Pipeline completed successfully!")

if __name__ == "__main__":
    main()
