import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from scipy.sparse import hstack
import numpy as np
import spacy
from preprocess import clean_text

# Load spaCy model with parser disabled to save memory
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm', disable=['parser'])

def add_sentiment_score(df, text_column='text'):
    """
    Add a sentiment polarity score column to the dataframe using TextBlob.
    """
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    return df

def add_additional_features(df, text_column='text'):
    """
    Add additional features: text length, punctuation count, uppercase ratio, word count, average word length, stopword ratio, num sentences, unique word ratio, has exclamation, has question, POS tags, named entities.
    """
    df['text_length'] = df[text_column].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['punct_count'] = df[text_column].apply(lambda x: sum(1 for c in x if c in '!?') if isinstance(x, str) else 0)
    df['upper_ratio'] = df[text_column].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if isinstance(x, str) and len(x) > 0 else 0)
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['avg_word_length'] = df[text_column].apply(lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) and x.split() else 0)
    df['stopword_ratio'] = df[text_column].apply(lambda x: len([word for word in x.lower().split() if word in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]) / len(x.split()) if isinstance(x, str) and x.split() else 0)
    df['num_sentences'] = df[text_column].apply(lambda x: len([s for s in x.split('.') if s.strip()]) if isinstance(x, str) else 0)
    df['unique_word_ratio'] = df[text_column].apply(lambda x: len(set(x.lower().split())) / len(x.split()) if isinstance(x, str) and x.split() else 0)
    df['has_exclamation'] = df[text_column].apply(lambda x: 1 if isinstance(x, str) and '!' in x else 0)
    df['has_question'] = df[text_column].apply(lambda x: 1 if isinstance(x, str) and '?' in x else 0)

    # Advanced features using spaCy
    df['num_nouns'] = df[text_column].apply(lambda x: len([token for token in nlp(x) if token.pos_ == 'NOUN']) if isinstance(x, str) else 0)
    df['num_verbs'] = df[text_column].apply(lambda x: len([token for token in nlp(x) if token.pos_ == 'VERB']) if isinstance(x, str) else 0)
    df['num_entities'] = df[text_column].apply(lambda x: len(nlp(x).ents) if isinstance(x, str) else 0)
    df['num_person_entities'] = df[text_column].apply(lambda x: len([ent for ent in nlp(x).ents if ent.label_ == 'PERSON']) if isinstance(x, str) else 0)
    df['num_org_entities'] = df[text_column].apply(lambda x: len([ent for ent in nlp(x).ents if ent.label_ == 'ORG']) if isinstance(x, str) else 0)
    df['num_gpe_entities'] = df[text_column].apply(lambda x: len([ent for ent in nlp(x).ents if ent.label_ == 'GPE']) if isinstance(x, str) else 0)

    return df

def tfidf_vectorize(df, text_column='text_clean', max_features=5000):
    """
    Apply TF-IDF vectorization on the cleaned text column.
    Returns the TF-IDF feature matrix and the vectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_tfidf = vectorizer.fit_transform(df[text_column].fillna(''))
    return X_tfidf, vectorizer

def combine_features(X_tfidf, df):
    """
    Combine TF-IDF features with numerical features (sentiment, text_length, punct_count, upper_ratio, word_count, avg_word_length, stopword_ratio, num_sentences, unique_word_ratio, has_exclamation, has_question, num_nouns, num_verbs, num_entities, num_person_entities, num_org_entities, num_gpe_entities) into a single feature matrix.
    """
    numerical_features = df[['sentiment', 'text_length', 'punct_count', 'upper_ratio', 'word_count', 'avg_word_length', 'stopword_ratio', 'num_sentences', 'unique_word_ratio', 'has_exclamation', 'has_question', 'num_nouns', 'num_verbs', 'num_entities', 'num_person_entities', 'num_org_entities', 'num_gpe_entities']].values
    scaler = MinMaxScaler()
    numerical_scaled = scaler.fit_transform(numerical_features)
    X_combined = hstack([X_tfidf, numerical_scaled])
    return X_combined, scaler

def predict_fake_news(text, vectorizer, model, scaler):
    """
    Predict whether the given text is real or fake news.
    """
    # Clean and preprocess the text
    text_clean = clean_text(text)

    # Calculate sentiment and additional features
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

    # Transform text using the loaded vectorizer
    X_input = vectorizer.transform([text_clean])

    # Get top 3 TF-IDF terms
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X_input.toarray()[0]
    top_indices = tfidf_scores.argsort()[-3:][::-1]  # Top 3 indices, descending
    top_tfidf_terms = [(feature_names[i], round(tfidf_scores[i], 4)) for i in top_indices if tfidf_scores[i] > 0]
    if not top_tfidf_terms:
        top_tfidf_terms = []  # Fallback if no significant terms

    # Scale numerical features using the loaded scaler
    numerical_features = [[sentiment, text_length, punct_count, upper_ratio, word_count, avg_word_length, stopword_ratio, num_sentences, unique_word_ratio, has_exclamation, has_question, num_nouns, num_verbs, num_entities, num_person_entities, num_org_entities, num_gpe_entities]]
    numerical_scaled = scaler.transform(numerical_features)

    # Combine text features and numerical features
    X_combined = hstack([X_input, numerical_scaled])

    # Make prediction
    pred = model.predict(X_combined)[0]
    prediction = "Fake" if pred == 1 else "Real"

    # Calculate confidence if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_combined)[0]
        confidence = round(100 * max(proba), 2)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_combined)[0]
        confidence = round(100 * (1 / (1 + np.exp(-decision))), 2)  # Sigmoid of decision function
    else:
        confidence = None

    return prediction, confidence, sentiment, top_tfidf_terms
