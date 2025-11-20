import os
import pickle
from flask import Flask, request, render_template_string

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    loaded = True
else:
    loaded = False

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template_string(open('templates/index.html').read())

# ...existing code for Flask routes...