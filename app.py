import streamlit as st
import joblib
import spacy

# 🔄 Load spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# 🧹 Clean and prepare text
def clean_input(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# 📦 Load trained ML components
model = joblib.load(r"climate_news_model.pkl")
vectorizer = joblib.load(r"climate_news_vectorizer.pkl")

# 🔍 Predict category
def classify_article(article):
    cleaned = clean_input(article)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

# 🌐 App Layout and Design
st.set_page_config(page_title="Climate News Categorizer", layout="centered")

st.markdown("""
    <style>
        .block-container {
            max-width: 750px;
            margin: auto;
            padding-top: 40px;
        }
        .category-box {
            font-size: 18px;
            font-weight: 600;
            padding: 10px;
            margin-top: 20px;
            text-align: center;
            border-radius: 10px;
            background-color: #eaf2f8;
            color: black;
        }
        .footer {
            font-size: 12px;
            color: #888;
            text-align: right;
        }
        textarea {
            border-radius: 10px !important;
            padding: 12px !important;
            border: 1px solid #bbb !important;
        }
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            background-color: #2e86de;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #1b4f72;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Climate News Categorizer")

user_input = st.text_area("Paste your climate-related article below:", placeholder="Example: 'UN urges immediate action as global temperatures hit new record.'")

st.markdown(f"<p class='footer'>Character count: {len(user_input)}</p>", unsafe_allow_html=True)

if st.button("Predict Category"):
    if user_input.strip():
        predicted = classify_article(user_input)
        st.markdown(f'<div class="category-box">🔎 Predicted Category: <strong>{predicted}</strong></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
