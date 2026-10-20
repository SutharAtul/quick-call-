
import os
import httpx
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE")
COLLECTION_NAME = os.getenv(
    "MONGODB_COLLECTION", "ai_knowledge_chunks"
)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv(
    "VOYAGE_MODEL", "voyage-multimodal-3.5"
)
VECTOR_INDEX_NAME = os.getenv(
    "VECTOR_INDEX_NAME", "knowledge_vector_index"
)

import certifi

if not all([
    MONGODB_URI,
    DATABASE_NAME,
    VOYAGE_API_KEY,
]):
    raise RuntimeError(
        "MongoDB or VoyageAI configuration is missing."
    )

client = MongoClient(
    MONGODB_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=3000,
)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]



def embed_texts(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Generate Voyage multimodal embeddings for text with rate limit retry handling."""
    if not texts:
        return []

    payload = {
        "model": VOYAGE_MODEL,
        "inputs": [
            {
                "content": [
                    {"type": "text", "text": text}
                ]
            }
            for text in texts
        ],
    }

    for attempt in range(max_retries + 1):
        response = httpx.post(
            "https://api.voyageai.com/v1/multimodalembeddings",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )

        if response.status_code == 429:
            if attempt < max_retries:
                retry_after = response.headers.get("retry-after")
                wait_seconds = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 22
                )
                print(
                    f"\n[Voyage AI Rate Limit (429)] Waiting {wait_seconds}s "
                    f"before retry ({attempt + 1}/{max_retries})..."
                )
                import time
                time.sleep(wait_seconds)
                continue

        response.raise_for_status()
        break

    result = response.json()
    items = sorted(
        result["data"], key=lambda item: item["index"]
    )
    embeddings = [item["embedding"] for item in items]

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count did not match input.")

    if any(len(vector) != 1024 for vector in embeddings):
        raise RuntimeError(
            "Expected 1024-dimensional vectors. "
            "Check the Voyage model and Atlas index configuration."
        )

    return embeddings


def split_into_chunks(text: str, max_chars: int = 900) -> list[str]:
    """Simple paragraph-aware chunking for the initial prototype."""
    paragraphs = [
        paragraph.strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        candidate = (
            f"{current}\n{paragraph}" if current else paragraph
        )

        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def store_document(
    source_id: str,
    title: str,
    text: str,
) -> int:
    """Replace one source's chunks in the JusBid knowledge base."""
    chunks = split_into_chunks(text)
    vectors = embed_texts(chunks)

    documents = []
    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors)
    ):
        documents.append({
            "_id": f"{source_id}:{index}",
            "source_id": source_id,
            "title": title,
            "chunk_index": index,
            "text": chunk,
            "embedding": vector,
            "status": "approved",
        })

    # Remove stale chunks if a source is re-ingested.
    collection.delete_many({
        "source_id": source_id,
    })

    if documents:
        collection.insert_many(documents)

    return len(documents)


def store_documents_batch(
    documents: list[dict],
) -> int:
    """Batch embed and store multiple JusBid knowledge documents efficiently."""
    if not documents:
        return 0

    texts_to_embed = [
        f"{doc.get('title', '')}\n{doc.get('content') or doc.get('text', '')}\nKeywords: {', '.join(doc.get('keywords', []))}"
        for doc in documents
    ]

    vectors = embed_texts(texts_to_embed)

    mongo_docs = []
    for doc, vector in zip(documents, vectors):
        source_id = str(doc.get("doc_id") or doc.get("source_id") or doc.get("title"))
        doc_key = f"jusbid:{source_id}" if not source_id.startswith("jusbid:") else source_id
        mongo_docs.append({
            "_id": doc_key,
            "company_id": doc.get("company_id", "jusbid"),
            "source_id": source_id,
            "doc_id": source_id,
            "title": doc.get("title", ""),
            "category": doc.get("category", "general"),
            "text": doc.get("content") or doc.get("text", ""),
            "keywords": doc.get("keywords", []),
            "status": "approved",
            "version": doc.get("version", 1),
            "embedding": vector,
        })

    for mdoc in mongo_docs:
        collection.replace_one(
            {"_id": mdoc["_id"]},
            mdoc,
            upsert=True,
        )

    return len(mongo_docs)


import json
from pathlib import Path

LOCAL_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "jusbid_mongodb_fake_data" / "jusbid_mongodb_fake_data" / "knowledge_base.json"


def retrieve_local_fallback(question: str, limit: int = 4) -> list[dict]:
    """Fallback keyword search against local knowledge_base.json if MongoDB Atlas is unreachable."""
    if not LOCAL_JSON_PATH.exists():
        return []

    try:
        with open(LOCAL_JSON_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)

        q_lower = question.lower()
        q_words = set(w.strip("?,.!\"'") for w in q_lower.split() if len(w) > 2)

        scored = []
        for d in docs:
            title = d.get("title", "")
            content = d.get("content", "")
            keywords = d.get("keywords", [])
            text_corpus = f"{title} {content} {' '.join(keywords)}".lower()

            matches = sum(1 for w in q_words if w in text_corpus)
            if matches > 0:
                scored.append((matches, {
                    "source_id": d.get("doc_id", "kb"),
                    "title": title,
                    "text": content,
                    "category": d.get("category", ""),
                    "score": round(matches / max(len(q_words), 1), 3),
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]
    except Exception as e:
        print(f"Local fallback error: {e}")
        return []


def retrieve(
    question: str,
    limit: int = 4,
) -> list[dict]:
    """Retrieve approved chunks from the JusBid knowledge base with automatic local fallback."""
    try:
        query_vector = embed_texts([question])[0]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": limit,
                    "filter": {
                        "status": {"$eq": "approved"},
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "source_id": 1,
                    "title": 1,
                    "text": 1,
                    "category": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(collection.aggregate(pipeline))
        if results:
            return results
    except Exception as exc:
        print(f"[RAG Store] MongoDB Vector Search failed or timed out ({exc}). Using local knowledge fallback...")

    # Graceful fallback when MongoDB is unreachable
    return retrieve_local_fallback(question, limit=limit)



