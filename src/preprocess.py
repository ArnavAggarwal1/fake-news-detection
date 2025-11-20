import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """
    Preprocess the text: remove punctuation, numbers, convert to lowercase,
    remove stopwords, tokenize, and stem.
    """
    if not isinstance(text, str):
        return ""

    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except AttributeError as e:
        # If WordNet is corrupted or has issues, skip lemmatization
        print(f"Warning: Lemmatization failed due to {e}. Skipping lemmatization.")
        pass  # tokens remain as is

    # Join back to string
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='text'):
    """
    Apply clean_text to the text column of the dataframe.
    """
    df[text_column + '_clean'] = df[text_column].apply(clean_text)
    return df
