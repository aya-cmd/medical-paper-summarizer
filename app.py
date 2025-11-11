import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer


nltk.download('punkt')
nltk.download('stopwords')

#  PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# preprocessing data
def clean(text):
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens if w.isalpha()]
    return [w for w in words if not w in stopwords.words('english')]

#  Streamlit
st.title("📄 Medical Paper Summarizer 🩺")
st.write("medic paper  (PDF) .")

uploaded_file = st.file_uploader("Upload your medical paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("جارٍ استخراج النص من الملف..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    st.success("تم استخراج النص بنجاح ✅")

    if st.button("summary model"):
        with st.spinner("..."):
            #
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

            #  (embedding)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(clean(pdf_text[:2000]))

            # summarizer
            summary = summarizer(pdf_text[:2000], max_length=150, min_length=30, do_sample=False)
            st.subheader("📑summary:")
            st.write(summary[0]['summary_text'])

