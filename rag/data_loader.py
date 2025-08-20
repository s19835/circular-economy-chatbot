import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
import pandas as pd

#paths
raw_data_dir = Path("../data/raw")
processed_data_dir = Path("../data/processed")
processed_data_dir.mkdir(parents=True, exist_ok=True)

def load_pdfs(data_dir=raw_data_dir):
    """
    Load all the pdfs form the raw data directory.
    Return a list of langchain documents objects.
    """
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = data_dir / file
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} pages form {file}")
    return docs

def save_as_csv(docs, output_file=processed_data_dir / "documents.csv"):
    """
    Save extracted text as csv with page metadata
    """
    rows = []
    for d in docs:
        rows.append({
            "content": d.page_content,
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", -1)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved {len(df)} chunks to {output_file}")

if __name__ == "__main__":
    documents = load_pdfs()
    if documents:
        save_as_csv(documents)
    else:
        print("No PDFs found in data/raw. Please add some first.")