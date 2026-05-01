from model import ResumeModel
from utils import clean_text

model = ResumeModel()

def analyze(resume_text, job_text):
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    score = model.compute_similarity(resume_text, job_text)

    return {
        "score": score,
        "interpretation": interpret_score(score)
    }

def interpret_score(score):
    if score > 0.75:
        return "High alignment with the job description."
    elif score > 0.5:
        return "Moderate alignment. Improvements recommended."
    else:
        return "Low alignment. Resume likely needs tailoring."