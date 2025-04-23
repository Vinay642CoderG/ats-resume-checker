import re
import spacy
import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import time

# Load spaCy's English model once (do this globally)
nlp = spacy.load("en_core_web_sm")

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="ATS Resume Checker",
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
    """
    Extracts keywords from text, removing stop words, short words,
    verbs, common nouns (NOUN), and proper nouns (PROPN) using spaCy.

    Args:
        text (str): The input text.
        stop_words (set): A set of stop words to remove.

    Returns:
        list: A list of extracted keywords (lowercase).
    """
    # Process the text with spaCy
    doc = nlp(text)

    IGNORE_CUSTOM_KEYWORDS = ['structured',
                              'structure', 'preferred', 'meaningful']

    keywords = []
    # Define the Part-of-Speech tags we want to remove
    # 'VERB': Verbs
    # 'NOUN': Common nouns
    # 'PROPN': Proper nouns
    pos_to_remove = {"VERB", "NOUN", "PROPN"}

    for token in doc:
        # Convert to lowercase for consistent comparison
        token_text_lower = token.lower_

        # Filter out punctuation, spaces, and words that are too short
        if token.is_punct or token.is_space or len(token_text_lower) <= 3:
            continue

        # Filter out words based on their Part-of-Speech tag
        if token.pos_ in pos_to_remove:
            continue

        # Filter out stop words using the provided set
        if token_text_lower in ENGLISH_STOP_WORDS:
            continue

        # Filter out stop words using the provided set
        if token_text_lower in IGNORE_CUSTOM_KEYWORDS:
            continue

        # If the token passed all filters, add its lowercase text to the keywords list
        keywords.append(token_text_lower)

    # The original function used set() on the initial split,
    # implicitly making keywords unique. Let's maintain uniqueness.
    return list(set(keywords))


def rank_resumes(job_description, resumes, model):
    doc_embeddings = model.encode(
        [job_description] + resumes, convert_to_tensor=True).cpu().numpy()
    job_desc_vector = doc_embeddings[0]
    resume_vectors = doc_embeddings[1:]

    # Calculate cosine similarity using NumPy
    similarity_scores = np.inner(resume_vectors, job_desc_vector) / (
        np.linalg.norm(resume_vectors, axis=1) *
        np.linalg.norm(job_desc_vector)
    )

    return similarity_scores.tolist()


def generate_resume_tips(score, resume_text, job_keywords):
    # Handle empty or non-list job_keywords
    if not isinstance(job_keywords, list) or not job_keywords:
        return "🔍 No job keywords provided. Focus on industry-specific skills and quantifiable achievements."

    # Preprocess resume tokens (handle special characters, case-insensitive)
    doc_resume = nlp(resume_text.lower())
    # Removed is_alpha check
    resume_lemmas = {token.lemma_ for token in doc_resume if not token.is_stop}

    # Process job keywords with deduplication and validation
    keyword_lemma_map = {}
    seen_keywords = set()

    for kw in job_keywords:
        # Skip non-strings and empty keywords
        if not isinstance(kw, str) or not kw.strip():
            continue

        # Case-insensitive deduplication
        kw_lower = kw.lower().strip()
        if kw_lower in seen_keywords:
            continue
        seen_keywords.add(kw_lower)

        # Lemmatize keyword
        kw_doc = nlp(kw_lower)
        if kw_doc:
            lemma = kw_doc[0].lemma_
            keyword_lemma_map[kw] = lemma  # Store original keyword with case

    # Find missing keywords (original casing)
    missing_keywords = [
        original_kw for original_kw, lemma in keyword_lemma_map.items()
        if lemma not in resume_lemmas
    ]

    # Grammar-optimized responses
    if score > 80:
        return "🔥 Excellent match! Your resume demonstrates strong keyword alignment and clarity."
    elif score > 60:
        if missing_keywords:
            keyword_list = ", ".join(f"'{kw}'" for kw in missing_keywords[:5])
            return (
                f"✅ Good match! Consider adding: {keyword_list}. "
                "Use variations like verbs/nouns (e.g., 'managed' and 'managing')."
            )
        else:
            return (
                "✅ Good match! Enhance impact by:\n"
                "- Adding metrics to achievements\n"
                "- Using industry-specific action verbs\n"
                "- Highlighting promotions/successes"
            )
    else:
        if missing_keywords:
            keyword_list = ", ".join(f"'{kw}'" for kw in missing_keywords[:5])
            return (
                f"⚡ Needs improvement! Critical missing keywords: {keyword_list}.\n"
                "Tips:\n"
                "- Mirror exact phrases from job description\n"
                "- Include both acronyms and full terms (e.g., 'SEO' and 'Search Engine Optimization')\n"
                "- Add sections Like: "
            )
        else:
            return (
                "⚡ Low match! Improve by:\n"
                "- Using more industry jargon\n"
                "- Adding certifications/licenses\n"
                "- Quantifying work experience (e.g., 'Increased sales by 42%')"
            )


def check_sections(resume_text):
    """
    Identifies potential missing standard sections in a resume.

    Args:
        resume_text (str): The text content of the resume.

    Returns:
        list: A list of capitalized section names that appear to be missing.
    """
    section_patterns = {
        "Skills": r"\b(skills|technical proficiencies|core competencies|technical skills)\b",
        "Experience": r"\b(experience|professional history|work history)\b",
        "Education": r"\b(education|academic background|qualifications)\b",
        "Projects": r"\b(projects|personal projects|portfolio)\b",
        "Certifications": r"\b(certifications|certification|licenses|credentials)\b",
    }

    resume_text_lower = resume_text.lower()
    missing_sections = []

    for section, pattern in section_patterns.items():
        if not re.search(pattern, resume_text_lower, re.IGNORECASE):
            missing_sections.append(section)

    return missing_sections

# --- Main App Logic ---


def main():
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
                tips += f" {', '.join(missing_sections)}."
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


if __name__ == "__main__":
    main()
# --- End of File ---
