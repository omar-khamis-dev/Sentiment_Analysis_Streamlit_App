# src/train.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # remove punctuation/numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# Load dataset
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Explorer', 'Tweets.csv'))
df = pd.read_csv(data_path)
df = df[['text', 'airline_sentiment']].dropna()
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['airline_sentiment']

# Compute class weights
classes = y.unique()
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight=class_weights))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(model_dir, exist_ok=True)
joblib.dump(pipeline, os.path.join(model_dir, 'sentiment_model.pkl'))
