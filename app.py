import streamlit as st
import os
import pandas as pd
from utils import extract_text_from_pdf, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

st.set_page_config(
    page_title="📄 AI Resume Screener",
    layout="centered",
    page_icon="🧠"
)

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    h1, h2, h3 {color: #30475e;}
    .stButton>button {
        background-color: #008cba;
        color: white;
        font-weight: bold;
    }
    .css-1v0mbdj {font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered Resume Screening Tool")
st.write("Upload multiple resumes and a job description to find the best match based on semantic similarity.")

uploaded_resumes = st.file_uploader("📎 Upload Resume PDFs", type="pdf", accept_multiple_files=True)
job_desc_file = st.file_uploader("📝 Upload Job Description (Text File)", type="txt")

def calculate_similarity(resume_texts, job_text):
    documents = [job_text] + resume_texts
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0]

def interpret_score(score):
    if score >= 0.75:
        return "🌟 Excellent"
    elif score >= 0.5:
        return "👍 Good"
    elif score >= 0.3:
        return "🤔 Average"
    else:
        return "❌ Low"

if uploaded_resumes and job_desc_file:
    job_description = job_desc_file.read().decode("utf-8")
    job_description = preprocess_text(job_description)

    resume_data = []
    for file in uploaded_resumes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        raw_text = extract_text_from_pdf(tmp_path)
        processed_text = preprocess_text(raw_text)
        resume_data.append({
            "filename": file.name,
            "raw_text": raw_text,
            "processed_text": processed_text
        })
        os.remove(tmp_path)

    texts = [r["processed_text"] for r in resume_data]
    scores = calculate_similarity(texts, job_description)

    results = []
    for i, res in enumerate(resume_data):
        results.append({
            "Resume": res["filename"],
            "Score": round(scores[i], 3),
            "Interpretation": interpret_score(scores[i])
        })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)

    st.subheader("📈 Matching Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="resume_rankings.csv",
        mime="text/csv"
    )