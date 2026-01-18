# app.pyi   mport necessary librariesl  ibraries
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection")
st.divider()
@st.cache_resource
def load_models():
    models = {
        'Decision Tree Classifier': joblib.load('trained_models/Decision Tree Classifier.pkl'),
        'Logistic Regression': joblib.load('trained_models/Logistic Regression.pkl')
        
    }
    vectorizer = joblib.load('trained_models/tfidf_vectorizer.pkl')
    return models, vectorizer
models, vectorizer = load_models()

model_name = st.selectbox("Select Model", options=models.keys())
st.divider()
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(
        word.lower()
        for word in text.split()
        if word.lower() not in stop_words
    )
    return text

text = st.text_area(
    "Enter the news text to analyze",
    placeholder="Type a news article or headline...",
    height=150
)

if st.button("Analyze the News"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = preprocess_text(text)
        vector = vectorizer.transform([cleaned_text])

        model = models[model_name]
        prediction = model.predict(vector)[0]
        st.divider()
        # Confidence
        confidence = model.predict_proba(vector).max() * 100

        if prediction == 1:
            st.success(f"🟢 **REAL NEWS** \n\nConfidence: **{confidence:.2f}%**")
        else:
            st.error(f"🔴 **FAKE NEWS** \n\nConfidence: **{confidence:.2f}%**")
