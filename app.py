mport nltk 
import fitz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from io import BytesIO
import os
import torch
nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_pdf(upload_file):
  try:
    with fitz.open(stream=upload_file.read(),filetype='pdf')as pdf:
      for pag in pdf:
        text+=pag.get_text()
  except Exception:
    for pag in pdf:
      text+= pag.get_text()
  return text

def cleac(text):
  tokens=word_tokenize(text)
  words=[w.lower for w in tokens if w.isalpha()]
  return [w for w in words if not w in stopwords.words('english')  ]


@st.cache_resource(show_spinner=False)    
def summarize_text(text, summary_length=200):
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    max_chunk = 1000
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summary_text = ""
    for chunk in text_chunks:
        summary_chunk = summarizer(chunk, max_length=summary_length, min_length=30, do_sample=False)
        summary_text += summary_chunk[0]['summary_text'] + " "
    return summary_text



def text_speech(text,lang='en'):
  tts=gTTS(text=text,lang=lang)
  audio_file=BytesIO()
  tts.write_to_fb(audio_file)
  audio_file.seek(0)
  return audio_file



    


st.set_page_config(page_title="📄 Medical Paper Summarizer + TTS", layout="wide")
st.title("📄 Medical Paper Summarizer & Text-to-Speech 🩺")
st.write("Upload a medical paper in PDF format — the app will extract the text, summarize it, and convert it to speech.")

# uploader
uploaded_file = st.file_uploader("Upload your medical paper (PDF)", type=["pdf"])

# user options
summary_length = st.slider('Choose the length of the summary (approx tokens)', 50, 500, 150)

# Option: instead of uploading, you can enter a local file path (for local testing)
use_local_path = st.checkbox("Use local PDF path instead of uploader (for testing)", value=False)
local_path = None
if use_local_path:
    local_path = st.text_input("Enter local file path (e.g. /content/13.pdf)")

if uploaded_file is not None or (use_local_path and local_path):
    # Extract text (supports both UploadedFile and local path)
    source = uploaded_file if uploaded_file is not None and not use_local_path else local_path
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(source)
    st.success("Text extracted successfully ✅")
    st.subheader("🔹 Paper Content (first 1000 characters)")
    st.write(pdf_text[:1000] + ("..." if len(pdf_text) > 1000 else ""))

    # summarize
    if st.button("Summarize Paper"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(pdf_text, summary_length=summary_length)
        st.subheader("📑 Summary:")
        st.write(summary)

        # text to speech
        with st.spinner("Converting summary to speech..."):
            audio_file = text_speech(summary, lang='en')
        st.audio(audio_file, format="audio/mp3")

        # download buttons
        st.download_button("Download Summary (TXT)", data=summary, file_name="summary.txt")
        st.download_button("Download Summary Audio (MP3)", data=audio_file, file_name="summary.mp3", mime="audio/mp3")
else:
    st.info("Upload a PDF file or enable 'Use local PDF path' to start.")

