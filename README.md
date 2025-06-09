# ✨ Sentiment Analysis

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![NLP](https://img.shields.io/badge/NLP-Scikit--learn-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/WebApp-Streamlit-red.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-yellow.svg)
![IDE-VSCode](https://img.shields.io/badge/IDE-VS%20Code-007ACC?logo=visualstudiocode)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

✅ The project was deployed as an interactive interface using Streamlit:
## 🌐 Live Demo
🔗 [sentiment-analysis-ai-project.streamlit.app](https://sentiment-analysis-ai-app.streamlit.app)

---

## 🧠 Overview
This project demonstrates a **machine learning pipeline for sentiment analysis** using real-world Twitter data related to airline customer feedback. The goal is to classify sentiments as `positive`, `neutral`, or `negative`, and display the prediction and confidence level in a web interface.

---

## 🎯 Project Highlights
- Built using `scikit-learn`, `nltk`, and `Streamlit`
- Trained on real-world airline tweets dataset from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Achieved ~78% accuracy
- Deployed as an interactive web app
- Future-ready: can be extended to WhatsApp, customer service, and multilingual analysis

---

## 🛠 Features
- User inputs any sentence in English
- Model classifies sentiment (`positive`, `neutral`, `negative`)
- Shows prediction **and** confidence %
- Clean and modern UI using Streamlit

---

## 🚀 Quick Start
1. Clone the repository:
```bash
git clone https://github.com/omar-khamis-dev/Sentiment_Analysis.git
cd Sentiment_Analysis
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the app:

```
streamlit run src/app.py
```

---

## 📦 File Structure
```
📁Sentiment_Analysis/
│
├──📁src/
│   ├── app.py                 # Streamlit web app
│   └── utils.py               # Utility functions
│
├──📁models/
│   └── sentiment_model.pkl    # Trained model
├───📁 Data_Explorer
│   ├── Tweets.csv                         # Raw dataset
│   └── database.sqlite                    # SQLite version of dataset
├── requirements.txt
└── README.md

```

---

## 📈 Model Performance (Sample)
```
          precision    recall  f1-score   support

negative       0.78      0.96      0.86      1889
 neutral       0.72      0.35      0.47       580
positive       0.82      0.55      0.66       459

accuracy                           0.78      2928

```

---

## 🧠 Future Enhancements
- Switch from CountVectorizer to TF-IDF or Word Embeddings
- Support Arabic + English classification
- Build API endpoint for real-time use
- Connect to chatbots, WhatsApp, or support systems
- Try deep learning models (BERT, LSTM)

---

## 👨‍💻 Author
## Omar Khamis
*AI & Robotics Enthusiast | Python Developer*

- 💼 [LinkedIn](https://www.linkedin.com/in/omar-khamis-dev)

- 💻 [GitHub](https://github.com/omar-khamis-dev)

- 📧 omar.khamis.dev@gmail.com

---

## 📜 License
Licensed under the `Apache 2.0` License.
