import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

#paths
data_file = "../data/processed/documents.csv"
index_file = "../data/processed/faiss_index.bin"
mapping_file = "../data/processed/mapping.csv"

def create_embeddings():
    #load data
    df = pd.read_csv(data_file)
    texts = df["content"].tolist()
    print(f"Loaded {len(texts)} text chunks")

    #load-embedding-model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    #create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))

    #save FAISS index
    faiss.write_index(index, index_file)
    df.to_csv(mapping_file, index=False)
    print(f"✅ Saved FAISS index to {index_file}")
    print(f"✅ Saved mapping file to {mapping_file}")

if __name__ == "__main__":
    if os.path.exists(data_file):
        create_embeddings()
    else:
        print("No processed documents found. Run data_loader.py first.")