import os
import sys
import pickle
import json
import logging
from flask import Flask, request, render_template, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering import predict_fake_news

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Load model, vectorizer, and scaler
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        loaded = True
        logging.info("Model, vectorizer, and scaler loaded successfully.")
    else:
        logging.warning("Model artifacts not found. Please run the training pipeline.")
        loaded = False
except Exception as e:
    logging.error(f"Error loading model artifacts: {e}")
    loaded = False

app = Flask(__name__, template_folder="templates")

logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    sentiment = None
    top_tfidf_terms = None
    if request.method == "POST":
        text = request.form.get("text", "")
        if not loaded:
            error = "Model not loaded. Please train and save the model first."
        elif not text.strip():
            error = "Please enter some news text."
        else:
            try:
                prediction, confidence, sentiment, top_tfidf_terms = predict_fake_news(text, vectorizer, model, scaler)
            except Exception as e:
                error = f"Error during prediction: {e}"
                sentiment = None
                top_tfidf_terms = None
    top_tfidf_terms_json = json.dumps(top_tfidf_terms) if top_tfidf_terms else '[]'
    return render_template("index.html", prediction=prediction, confidence=confidence, error=error, sentiment=sentiment, top_tfidf_terms=top_tfidf_terms, top_tfidf_terms_json=top_tfidf_terms_json)

@app.route("/welcome", methods=["GET"])
def welcome():
    logging.info(f"Request received: {request.method} {request.path}")
    return jsonify({"message": "Welcome to the Fake News Detection API!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
