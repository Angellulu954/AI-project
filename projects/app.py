import streamlit as st
from utils import extract_text_from_pdf
from backend import analyze

st.set_page_config(page_title="Resume Intelligence System", layout="wide")

st.title("Resume–Job Matching Intelligence System")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area("Job Description", height=300)

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)

    result = analyze(resume_text, job_description)

    st.subheader("Analysis Results")

    score = result["score"] * 100

    st.metric("Match Score", f"{score:.2f}%")

    st.progress(min(int(score), 100))

    st.write(result["interpretation"])