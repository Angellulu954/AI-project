from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_similarity(self, resume_text, job_text):
        embeddings = self.model.encode([resume_text, job_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(score)