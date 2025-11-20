# Fake News Detection

A machine learning project for detecting fake news using natural language processing (NLP) techniques and multiple classification models. This project implements a comprehensive pipeline from data preprocessing to model deployment, including a Flask web application for real-time predictions.

## Features

- **Data Preprocessing**: Cleaning and preprocessing of news articles using NLTK and spaCy
- **Feature Engineering**: Advanced features including TF-IDF vectorization, POS tagging, and named entity recognition
- **Multiple Models**: Comparison of Logistic Regression, Naive Bayes, Random Forest, SVM, and XGBoost
- **Hyperparameter Tuning**: Grid search with cross-validation for optimal model performance
- **Web Application**: Flask-based web app for easy prediction interface
- **Visualization**: Performance metrics and ROC curves for model comparison

## Dataset

The project uses the "Fake and Real News Dataset" from Kaggle, containing over 44,000 news articles labeled as fake or real.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ArnavAggarwal1/fake-news-detection.git
cd fake-news-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Download the dataset:
```bash
python data/download_data.py
```

## Usage

### Training the Models
```bash
python notebooks/fake_news_detection.py
```

### Running the Web Application
```bash
python app/app.py
```
Navigate to `http://127.0.0.1:5000` in your browser.

### Jupyter Notebook
```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

## Project Structure

```
fake-news-detection/
├── app/                    # Flask web application
│   ├── templates/          # HTML templates
│   ├── model.pkl          # Trained model
│   ├── vectorizer.pkl     # TF-IDF vectorizer
│   └── scaler.pkl         # Feature scaler
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks and training scripts
│   └── plots/             # Generated plots
├── src/                    # Source code modules
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Results

The XGBoost model achieved the highest performance with:
- Accuracy: 95%
- F1-Score: 95%
- ROC-AUC: 0.98

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
