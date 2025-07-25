import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Ensure required NLTK resources
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

ps = PorterStemmer()

# ✅ Text preprocessing
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = [word for word in tokens if word.isalnum()]
    filtered = [ps.stem(word) for word in clean_tokens if word not in stopwords.words('english')]
    return " ".join(filtered)

# ✅ Load model & vectorizer
if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
    st.error("❌ Model or vectorizer file not found. Please train the model using `train_model.py`.")
else:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # ✅ Streamlit UI
    st.title("📩 Email/SMS Spam Classifier")
    input_sms = st.text_area("✉️ Enter the message")

    if st.button("Predict"):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message to classify.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            try:
                result = model.predict(vector_input)[0]
                if result == 1:
                    st.error("🚫 Spam")
                else:
                    st.success("✅ Not Spam")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
