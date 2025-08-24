import os
import pickle
from retrieval import search
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

cache_file = ".vector_cache.pkl"

# Initialize LLM
# LLM = ChatOpenAI(
#     temperature=0,
#     model='gpt-3.5-turbo',
#     api_key=os.environ.get("OPENAI_API_KEY")
# )

LLM = ChatOllama(
    model="mistral"
)

def load_cache():
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

vector_cache = load_cache()

def answer_questions(query: str, k: int=3, language: str='en'):
    """
    1. Retrieve top-k relevant chunks using retrieval.py
    2. Cache the embeddings to speed up repeated queries
    """
    # check cache first
    cache_key = f"{query}_{language}"
    if cache_key in vector_cache:
        hits = vector_cache[cache_key]
    else:
        hits = search(query, k)
        vector_cache[cache_key] = hits
        save_cache(vector_cache)
    
    if not hits:
        return "No relevant information found"
    
    #combine context
    context = "\n\n".join([hit.get('text', '') for hit in hits])

    system_prompt = (
        "You are an AI assistant providing answers strictly based on the provided context. "
        "Provide answers in the requested language only. "
        "If the answer is not in the context, say 'I don't know.'"
    )

    # response = LLM([
    #     SystemMessage(content=system_prompt),
    #     HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\nLanguage: {language}")
    # ])
    response = LLM.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\nLanguage: {language}")
    ])

    # HuggingFacePipeline returns a string or a dict, not an object with .content
    if isinstance(response, str):
        return response
    elif isinstance(response, dict) and "generated_text" in response:
        return response["generated_text"]
    else:
        return str(response)

if __name__ == "__main__":
    print("RAG-based QA chatbot (multilingual). Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ")
        if q.lower() == "exit":
            break
        lang = input("Language (default 'en'): ") or "en"
        ans = answer_questions(q, language=lang)
        print("\nAnswer:\n", ans, "\n")
