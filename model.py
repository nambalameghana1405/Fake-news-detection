import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Download stopwords if not already present
try:
    stopwords.words('english')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Text Preprocessing ---
def preprocess_text(text):
    """
    Cleans and preprocesses a given text string.
    - Converts to lowercase.
    - Removes punctuation, numbers, URLs, and extra whitespace.
    - Removes stopwords.
    """
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()
    # Remove text in brackets
    text = re.sub('\[.*?\]', '', text)
    # Remove non-alphanumeric characters and replace with space
    text = re.sub("\\W", " ", text)
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove newlines and multiple spaces
    text = re.sub('\n', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text


# --- Training Function ---
def train_and_save_model():
    """
    Loads data, trains a PassiveAggressiveClassifier model, and saves it.
    This function is a fallback if the model files are not found.
    """
    print("🤖 Training model from scratch...")
    try:
        fake_df = pd.read_csv('Fake.csv')
        # Updated filename to 'True2.csv' as requested
        true_df = pd.read_csv('True.csv')
    except FileNotFoundError:
        print("❌ Error: 'Fake.csv' or 'True2.csv' not found in this folder.")
        return False

    # Add labels to the dataframes
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine dataframes and shuffle
    df = pd.concat([fake_df, true_df]).reset_index(drop=True).dropna()
    df = df[['text', 'label']]
    
    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text)

    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Train the PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)
    print(f"✅ Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model and vectorizer
    try:
        joblib.dump(pac, 'model.pkl')
        joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')
        print("✅ Model and vectorizer saved successfully.")
        return True
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()
