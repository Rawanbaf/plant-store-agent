import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="plant_store_knowledge")

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


SYSTEM_PROMPT = """You are an internal assistant for plant store employees.
You answer questions using ONLY the context provided below. Do not make up information.

If the context includes past customer complaints, you can reference specific details (order numbers, statuses, issues).
If the context includes store policy, answer according to that policy.
If the context doesn't contain enough information to answer, say so honestly instead of guessing.

Keep answers concise and direct, written for an employee who needs a quick, useful answer."""


def complaint_text(c):
    return (
        f"Customer: {c['sender_email']}\n"
        f"Team: {c['team']}\n"
        f"Category: {c['category']}\n"
        f"Item: {c['item']}\n"
        f"Order number: {c['order_number']}\n"
        f"Summary: {c['summary']}\n"
        f"Status: {c['status']}\n"
        f"Employee note: {c.get('employee_note', '')}\n"
        f"Full email: {c['full_email']}"
    )


def sync_complaints(complaints):
    if not complaints:
        return

    ids = [f"complaint_{c['id']}" for c in complaints]
    texts = [complaint_text(c) for c in complaints]
    metadatas = [{"source": "complaint", "complaint_id": c["id"]} for c in complaints]
    embeddings = get_embed_model().encode(texts).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


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


def find_similar(complaint, complaints, n=6):
    try:
        sync_complaints(complaints)

        query = f"{complaint['category']} {complaint['item']} {complaint['summary']}"
        query_embedding = get_embed_model().encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n + 1, len(complaints)),
            where={"source": "complaint"},
        )

        by_id = {c["id"]: c for c in complaints}
        similar = []
        for meta in results["metadatas"][0]:
            cid = meta.get("complaint_id")
            if cid and cid != complaint["id"] and cid in by_id:
                similar.append(by_id[cid])

        return similar
    except Exception as e:
        print("Similar tickets lookup failed:", e)
        return []