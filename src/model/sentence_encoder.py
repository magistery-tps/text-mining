from sentence_transformers import SentenceTransformer


class SentenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def __call__(self, sentences): 
        embeddings = self.model.encode(sentences)
        return {s: e  for s, e in zip(sentences, embeddings)}