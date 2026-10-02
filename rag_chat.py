import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="plant_store_knowledge")

SYSTEM_PROMPT = """You are an internal assistant for plant store employees.
You answer questions using ONLY the context provided below. Do not make up information.

If the context includes past customer complaints, you can reference specific details (order numbers, statuses, issues).
If the context includes store policy, answer according to that policy.
If the context doesn't contain enough information to answer, say so honestly instead of guessing.

Keep answers concise and direct, written for an employee who needs a quick, useful answer."""


def ask_question(question: str, n_results: int = 5) -> str:
    query_embedding = get_embed_model().encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n---\n\n".join(
        f"[{src}]\n{chunk}" for chunk, src in zip(retrieved_chunks, sources)
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response.output_text


