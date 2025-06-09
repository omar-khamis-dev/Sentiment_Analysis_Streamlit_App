import streamlit as st
from utils import clean_text, load_model

# تحميل النموذج
@st.cache_resource
def load_sentiment_model():
    model_path = '../models/sentiment_model.pkl'
    return load_model(model_path)

def main():
    st.title("Sentiment Analysis")

    model = load_sentiment_model()

    user_input = st.text_area("Enter text to specify sentiment:", height=150)

    if st.button("Analysis"):
        if user_input.strip() == "":
            st.warning("Please enter text for analysis.")
        else:
            cleaned = clean_text(user_input)

            proba = model.predict_proba([cleaned])[0]
            classes = model.classes_
            top_index = proba.argmax()
            prediction = classes[top_index]
            confidence = proba[top_index] * 100

            st.markdown(f"### Classification: :blue[{prediction}]")
            st.markdown(f"Confidence Percentage: :green[{confidence:.2f}%]")

if __name__ == "__main__":
    main()
