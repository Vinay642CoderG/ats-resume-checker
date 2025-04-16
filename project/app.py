import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import time

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Resume Ranking System",
                   page_icon="🚀", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: white; }
    .stApp { background-color: #1E1E1E; }
    h1 { text-align: center; color: #4CAF50; font-size: 36px; font-weight: bold; }
    .stTextArea, .stFileUploader {
        border-radius: 10px;
        box-shadow: 0 0 10px #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 18px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Ats Resume Checker")

col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=[
                                      "pdf"], accept_multiple_files=True)

with col2:
    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description...")

# --- Helper Functions ---


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return ""


def extract_keywords(text):
    words = set(text.lower().split())
    keywords = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 3]
    return keywords


def rank_resumes(job_description, resumes, model):
    doc_embeddings = model.encode([job_description] + resumes)
    job_desc_vector = doc_embeddings[0]
    resume_vectors = doc_embeddings[1:]
    scores = util.cos_sim(job_desc_vector, resume_vectors)[0].cpu().numpy()
    return scores


def generate_resume_tips(score, resume_text, job_keywords):
    resume_words = set(resume_text.lower().split())
    missing = [kw for kw in job_keywords if kw not in resume_words]
    if score > 80:
        return "🔥 Excellent match! Your resume is well-optimized."
    elif score > 60:
        if missing:
            return f"✅ Good match! Consider adding these keywords: {', '.join(missing[:5])}"
        else:
            return "✅ Good match! Maybe elaborate on your achievements."
    else:
        if missing:
            return f"⚡ Low match! Add skills like: {', '.join(missing[:5])}."
        else:
            return "⚡ Low match! Try improving your skills section and adding industry-specific terms."


def check_sections(resume_text):
    sections = ["skills", "experience",
                "education", "projects", "certifications"]
    missing_sections = [sec.capitalize()
                        for sec in sections if sec not in resume_text.lower()]
    return missing_sections

# --- Main App Logic ---


if uploaded_files and job_description:
    st.header("📊 Resume Rankings")

    # Extract text from resumes
    resumes = []
    for file in uploaded_files:
        resumes.append(extract_text_from_pdf(file))

    # Progress Bar
    progress_bar = st.progress(0)
    for i in range(50):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Load model and rank resumes
    model = load_model()
    scores = rank_resumes(job_description, resumes, model)
    job_keywords = extract_keywords(job_description)

    # Build results DataFrame
    results = []
    for i, file in enumerate(uploaded_files):
        score = scores[i] * 100
        resume_text = resumes[i]
        tips = generate_resume_tips(score, resume_text, job_keywords)
        missing_sections = check_sections(resume_text)
        if missing_sections:
            tips += f" Consider adding sections: {', '.join(missing_sections)}."
        results.append({
            "Resume": file.name,
            "Match Score (%)": round(score, 2),
            "AI Suggestion": tips
        })

    results_df = pd.DataFrame(results).sort_values(
        by="Match Score (%)", ascending=False)

    progress_bar.progress(100)

    # Display DataFrame
    st.dataframe(results_df.style.format(
        {"Match Score (%)": "{:.2f}"}), use_container_width=True)
    st.success(
        f"✅ Ranking Complete! 🎯 Top Match: {results_df.iloc[0]['Resume']}")

    # Bar Chart
    st.subheader("📈 Match Score Comparison")
    st.bar_chart(results_df.set_index("Resume")["Match Score (%)"])

    # Download Button
    st.download_button(
        "Download Results as CSV",
        results_df.to_csv(index=False),
        file_name="resume_ranking_results.csv",
        mime="text/csv"
    )

    # Resume Previews
    with st.expander("🔎 Preview Resumes"):
        for i, file in enumerate(uploaded_files):
            st.markdown(f"{file.name}")
            st.text_area("Preview", resumes[i]
                         [:1500], height=200, key=file.name)

elif uploaded_files and not job_description:
    st.info("Please enter a job description to start ranking.")

elif job_description and not uploaded_files:
    st.info("Please upload at least one PDF resume to start ranking.")

else:
    st.info("Upload resumes and enter a job description to begin.")

# --- End of File ---
