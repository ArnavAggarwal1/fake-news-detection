import pickle
import os
import sys
import numpy as np

# Add src to path
sys.path.append('src')

from feature_engineering import predict_fake_news
from preprocess import clean_text
from textblob import TextBlob
import spacy

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm', disable=['parser'])

# Load model, vectorizer, and scaler
MODEL_PATH = 'app/model.pkl'
VECTORIZER_PATH = 'app/vectorizer.pkl'
SCALER_PATH = 'app/scaler.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Test texts with expected labels
test_samples = [
    ("NASA’s Parker Solar Probe successfully entered the Sun’s outer atmosphere, the corona, collecting unprecedented data on solar winds.", "Real"),
    ("Breaking: Aliens have landed in New York City and are demanding to speak to the President!", "Fake"),
    ("The stock market closed higher today with gains in technology stocks.", "Real"),
    ("Shocking: Eating chocolate every day cures all diseases, scientists confirm!", "Fake"),
    ("The government announced new tax reforms to help middle-class families.", "Real"),
    ("Exclusive: Time travel is now possible with this new app!", "Fake"),
    ("The United Nations reported a decline in global poverty rates.", "Real"),
    ("Scientists discovered a new planet in our solar system.", "Real")
]

# Clean text
text_clean = clean_text(test_text)

# Calculate features manually to debug
sentiment = TextBlob(text_clean).sentiment.polarity
text_length = len(text_clean)
punct_count = sum(1 for c in text_clean if c in '!?')
upper_ratio = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if len(text_clean) > 0 else 0
word_count = len(text_clean.split())
avg_word_length = np.mean([len(word) for word in text_clean.split()]) if text_clean.split() else 0
stopword_ratio = len([word for word in text_clean.lower().split() if word in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]) / len(text_clean.split()) if text_clean.split() else 0
num_sentences = len([s for s in text_clean.split('.') if s.strip()])
unique_word_ratio = len(set(text_clean.lower().split())) / len(text_clean.split()) if text_clean.split() else 0
has_exclamation = 1 if '!' in text_clean else 0
has_question = 1 if '?' in text_clean else 0

# Advanced features using spaCy
doc = nlp(text_clean)
num_nouns = len([token for token in doc if token.pos_ == 'NOUN'])
num_verbs = len([token for token in doc if token.pos_ == 'VERB'])
num_entities = len(doc.ents)
num_person_entities = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
num_org_entities = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
num_gpe_entities = len([ent for ent in doc.ents if ent.label_ == 'GPE'])

print("Cleaned Text:", text_clean)
print("Sentiment:", sentiment)
print("Text Length:", text_length)
print("Punct Count:", punct_count)
print("Upper Ratio:", upper_ratio)
print("Word Count:", word_count)
print("Avg Word Length:", avg_word_length)
print("Stopword Ratio:", stopword_ratio)
print("Num Sentences:", num_sentences)
print("Unique Word Ratio:", unique_word_ratio)
print("Has Exclamation:", has_exclamation)
print("Has Question:", has_question)
print("Num Nouns:", num_nouns)
print("Num Verbs:", num_verbs)
print("Num Entities:", num_entities)
print("Num Person Entities:", num_person_entities)
print("Num Org Entities:", num_org_entities)
print("Num GPE Entities:", num_gpe_entities)

# TF-IDF
X_input = vectorizer.transform([text_clean])
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X_input.toarray()[0]
top_indices = tfidf_scores.argsort()[-10:][::-1]  # Top 10
top_tfidf_terms = [(feature_names[i], round(tfidf_scores[i], 4)) for i in top_indices if tfidf_scores[i] > 0]
print("Top TF-IDF Terms:", top_tfidf_terms)

# Scaled numerical features (all 17 features used in training)
numerical_features = [[sentiment, text_length, punct_count, upper_ratio, word_count, avg_word_length, stopword_ratio, num_sentences, unique_word_ratio, has_exclamation, has_question, num_nouns, num_verbs, num_entities, num_person_entities, num_org_entities, num_gpe_entities]]
numerical_scaled = scaler.transform(numerical_features)
print("Scaled Numerical Features:", numerical_scaled.flatten())

# Combined
X_combined = np.hstack([X_input.toarray(), numerical_scaled])
print("Combined Feature Shape:", X_combined.shape)

for test_text, expected in test_samples:
    print(f"\nTesting: {test_text[:50]}...")
    prediction, confidence, sentiment, top_tfidf_terms = predict_fake_news(test_text, vectorizer, model, scaler)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence}%")
    print(f"Sentiment: {sentiment}")
    print(f"Top TF-IDF Terms: {top_tfidf_terms}")
    print(f"Expected: {expected}, Correct: {prediction.lower() == expected.lower()}")
