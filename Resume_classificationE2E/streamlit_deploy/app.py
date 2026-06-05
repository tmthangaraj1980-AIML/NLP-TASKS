import streamlit as st
import joblib
import PyPDF2
from docx import Document

# Load model and vectorizer
model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# --- Sidebar ---
st.sidebar.title("Resume Classification Project")
st.sidebar.markdown("**Developed by Thangaraj** – AI/ML Intern (NLP Project)")

# --- Main Title ---
st.title("Resume Classification App")

# --- Option 1: Copy-paste text ---
resume_text = st.text_area("Paste Resume Text Here")

# --- Option 2: Upload PDF ---
uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
if uploaded_pdf is not None:
    reader = PyPDF2.PdfReader(uploaded_pdf)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    resume_text = pdf_text
    st.text_area("Extracted Resume Text (PDF)", resume_text, height=200)

# --- Option 3: Upload Word ---
uploaded_docx = st.file_uploader("Upload Resume (Word)", type=["docx"])
if uploaded_docx is not None:
    doc = Document(uploaded_docx)
    doc_text = "\n".join([para.text for para in doc.paragraphs])
    resume_text = doc_text
    st.text_area("Extracted Resume Text (Word)", resume_text, height=200)

# --- Classification ---
if st.button("Classify"):
    if resume_text.strip():
        transformed = tfidf.transform([resume_text])
        prediction = model.predict(transformed)[0]
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please provide resume text (paste or upload).")
