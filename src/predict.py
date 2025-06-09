# src/predict.py
from utils import load_model, clean_text

def predict_sentiment(text, model_path='models/sentiment_model.pkl', vectorizer_path='models/vectorizer.pkl'):
    model = load_model(model_path)
    vectorizer = load_model(vectorizer_path)

    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)
    return prediction[0]

# مثال تشغيل
if __name__ == '__main__':
    sample = "I love this product! It’s amazing."
    result = predict_sentiment(sample)
    print(f"Sentiment: {result}")
