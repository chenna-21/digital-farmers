import pandas as pd
import os
import sys

# Load dataset
# Load dataset
if os.path.exists("dataset.csv"):
    DATASET_PATH = "dataset.csv"
else:
    DATASET_PATH = "../dataset.csv"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGEngine:
    def __init__(self):
        print("Loading Knowledge Base...")
        try:
            self.df = pd.read_csv(DATASET_PATH)
            print(f"DEBUG: Dataset loaded. Shape: {self.df.shape}")
            print(f"DEBUG: First 5 rows:\n{self.df.head()}")
            # Basic cleaning
            self.df['combined_text'] = self.df['QueryText'].fillna('') + " " + self.df['KccAns'].fillna('')
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.df = pd.DataFrame(columns=['QueryText', 'KccAns', 'combined_text'])

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if self.df.empty:
            return

        print("Building Index (TF-IDF)...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['QueryText'].fillna(''))
        print("Index Built Successfully.")

    def search(self, query: str, k: int = 3):
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        
        # Compute Cosine Similarity
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k (descending order)
        related_docs_indices = cosine_similarities.argsort()[:-k-1:-1]
        
        results = []
        for idx in related_docs_indices:
            score = cosine_similarities[idx]
            if score > 0.0: # Only return if there is some similarity
                row = self.df.iloc[idx]
                results.append({
                    "question": row['QueryText'],
                    "answer": row['KccAns'],
                    "score": float(score)
                })
        return results

    def generate_response(self, query: str):
        # Retrieve context
        retrieved = self.search(query)
        
        if not retrieved:
            return "I'm sorry, I couldn't find any relevant information in my database."

        # Construct Context
        context_str = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved])
        
        # In a real scenario, call LLM here.
        # For now, if no API key, return a constructed answer based on top match.
        
        # Placeholder for OpenAI Logic
        # if os.getenv("OPENAI_API_KEY"):
        #     return call_openai(query, context_str)
            
        # Fallback / Mock Response logic for demo
        best_match = retrieved[0]
        return f"Based on similar queries, here is the information:\n\n**{best_match['answer']}**\n\n(Context: This was answer to '{best_match['question']}')"

rag_engine = RAGEngine()
