
import streamlit as st
import pickle, re, string

with open("language_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def detect_code_mixing(text):
    tokens = text.split()
    token_labels = []
    for token in tokens:
        cleaned = preprocess(token)
        if not cleaned:
            continue
        features = vectorizer.transform([cleaned])
        lang = model.predict(features)[0]
        token_labels.append((token, lang))

    if not token_labels:
        return None

    langs = [l for _, l in token_labels]
    en_ratio = langs.count("English") / len(langs)
    sw_ratio = langs.count("Swahili") / len(langs)

    if en_ratio > 0.15 and sw_ratio > 0.15:
        label = "Code-Mixed (English + Swahili)"
    elif en_ratio >= sw_ratio:
        label = "English"
    else:
        label = "Swahili"

    return label, en_ratio, sw_ratio, token_labels

st.set_page_config(page_title="Language ID + Code-Mixing", page_icon="🌍")
st.title("🌍 Language Identification with Code-Mixing Detection")

user_input = st.text_area("Enter text:", height=150,
    placeholder="Try: 'Ninaenda to the market kila siku'")

if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        result = detect_code_mixing(user_input)
        if result:
            label, en_ratio, sw_ratio, token_labels = result
            st.success(f"Detected: **{label}**")
            st.progress(en_ratio, text=f"English: {en_ratio:.0%}")
            st.progress(sw_ratio, text=f"Swahili: {sw_ratio:.0%}")

            st.markdown("#### Word-by-word breakdown")
            cols = st.columns(4)
            for i, (word, lang) in enumerate(token_labels):
                flag = "🇬🇧" if lang == "English" else "🇰🇪"
                cols[i % 4].markdown(f"**{word}** {flag}")

st.markdown("---")
st.caption("CSC423 NLP Term Project")
