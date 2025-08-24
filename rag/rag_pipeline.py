from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load embeddings + FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. Setup retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. Load Ollama model
llm = ChatOllama(model="mistral")  # change to llama3 if you prefer

# 4. RAG QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def rag_answer(question: str) -> str:
    """Answer using RAG pipeline"""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = "\n".join([doc.metadata.get("source", "N/A") for doc in result["source_documents"]])
    return f"**Answer:** {answer}\n\n**Sources:**\n{sources}"
