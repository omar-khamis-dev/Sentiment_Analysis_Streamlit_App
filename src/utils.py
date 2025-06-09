# src/utils.py
import re
import os
import joblib

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    return text.lower()

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    base_dir = os.path.dirname(__file__)  # مسار مجلد src
    model_path = os.path.abspath(os.path.join(base_dir, '..', 'models', 'sentiment_model.pkl'))
    return joblib.load(model_path)
