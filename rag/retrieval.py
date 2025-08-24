import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

#paths
index_file = "../data/processed/faiss_index.bin"
mapping_file = "../data/processed/mapping.csv"

def search(query, k=3):
    # load faiss index
    index = faiss.read_index(index_file)
    df = pd.read_csv(mapping_file)

    # encode query
    model = SentenceTransformer("all-MiniLM-L6-v2")  # <-- moved here
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")

    # search
    distances, indices = index.search(q_emb, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(df):
            results.append({
                "text": df.iloc[idx]["content"],
                "source": df.iloc[idx]["source"],
                "page": df.iloc[idx]["page"],
                "score": float(dist)
            })
    return results

if __name__ == "__main__":
    query = "What are circular economy business models?"
    hits = search(query, k=3)
    print("query:", query)
    for h in hits:
        print(f"📄 Page {h['page']} | {h['source']} | Score: {h['score']:.4f}")
        print(h["text"][:300], "...\n")
    
    os._exit(0)