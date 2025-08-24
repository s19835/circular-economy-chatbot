# Circular Economy Chatbot

A research-focused chatbot that answers queries about circular economy concepts using a Retrieval-Augmented Generation (RAG) approach.

## Features

- Answering questions based on curated PDF and CSV knowledge sources.
- Semantic search for efficient retrieval of relevant content.
- Supports both simple and advanced question-answering.
- Designed to integrate with a web UI (Gradio/Streamlit) for interactive use.

## Tech Stack

- Python, LangChain, FAISS, LlamaIndex
- OpenAI API for text embeddings and LLM responses
- Pandas, Numpy for data processing
- Optional: Gradio/Streamlit for UI

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare knowledge base documents (PDFs, CSVs) in the `data/` folder.
3. Run the QA script:

   ```bash
   python rag/qa_advanced.py
   ```

4. (Optional) Launch web UI for interactive use:

   ```bash
   streamlit run app.py
   ```

## Future Work

- Add support for multi-language content.
- Deploy as a Docker container with API endpoints.
- Integrate user feedback loop for improved answers.
