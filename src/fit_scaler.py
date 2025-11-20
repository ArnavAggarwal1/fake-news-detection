import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from preprocess import preprocess_dataframe
from feature_engineering import add_sentiment_score, add_additional_features, tfidf_vectorize

# Load data
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df], ignore_index=True)

# Sample a subset to avoid memory issues
df_sample = df.sample(n=10, random_state=42)

# Preprocess
df_sample = preprocess_dataframe(df_sample, 'text')

# Feature engineering
df_sample = add_sentiment_score(df_sample, 'text')
df_sample = add_additional_features(df_sample, 'text')
X_tfidf, vectorizer = tfidf_vectorize(df_sample, 'text_clean')

# Fit scaler on numerical features
numerical_features = df_sample[['sentiment', 'text_length', 'punct_count', 'upper_ratio', 'word_count', 'avg_word_length', 'stopword_ratio', 'num_sentences', 'unique_word_ratio', 'has_exclamation', 'has_question', 'num_nouns', 'num_verbs', 'num_entities', 'num_person_entities', 'num_org_entities', 'num_gpe_entities']].values
scaler = MinMaxScaler()
scaler.fit(numerical_features)

# Save scaler
import pickle
pickle.dump(scaler, open('app/scaler.pkl', 'wb'))
print("Scaler fitted and saved.")
print(f'Number of features: {scaler.n_features_in_}')
