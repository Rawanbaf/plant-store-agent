import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="plant_store_knowledge")


def chunk_knowledge_base(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("\n\n")
    chunks = []
    for section in sections:
        section = section.strip()
        if section:
            chunks.append(section)
    return chunks


def index_knowledge_base():
    chunks = chunk_knowledge_base("knowledge_base.txt")

    ids = [f"kb_{i}" for i in range(len(chunks))]
    embeddings = model.encode(chunks).tolist()
    metadatas = [{"source": "knowledge_base"} for _ in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"Indexed {len(chunks)} knowledge base chunks.")


def index_complaints():
    if not os.path.exists("complaints.json"):
        print("No complaints.json found, skipping.")
        return

    with open("complaints.json", "r", encoding="utf-8") as f:
        complaints = json.load(f)

    if not complaints:
        print("No complaints to index.")
        return

    ids = []
    texts = []
    metadatas = []

    for c in complaints:
        text = (
            f"Customer: {c['sender_email']}\n"
            f"Team: {c['team']}\n"
            f"Category: {c['category']}\n"
            f"Item: {c['item']}\n"
            f"Order number: {c['order_number']}\n"
            f"Summary: {c['summary']}\n"
            f"Status: {c['status']}\n"
            f"Full email: {c['full_email']}"
        )
        ids.append(f"complaint_{c['id']}")
        texts.append(text)
        metadatas.append({"source": "complaint", "complaint_id": c["id"]})

    embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Indexed {len(texts)} complaints.")


if __name__ == "__main__":
    index_knowledge_base()
    index_complaints()