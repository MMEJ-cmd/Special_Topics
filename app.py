
import streamlit as st
import pickle

# Load model and vectorizer
with open("language_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

import re, string

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Page config
st.set_page_config(page_title="Language Identifier", page_icon="🌍")

st.title("🌍 Language Identification System")
st.write("Detects whether a text is written in **English** or **Swahili**")

user_input = st.text_area("Enter your text below:", height=150, placeholder="Type or paste text here...")

if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = preprocess(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        if prediction == "English":
            st.success(f"🇬🇧 Detected Language: **{prediction}**")
        elif prediction == "Swahili":
            st.success(f"🇰🇪 Detected Language: **{prediction}**")

st.markdown("---")
st.caption("CSC423 NLP Term Project")
