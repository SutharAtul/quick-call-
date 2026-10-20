
import json
import os
from pathlib import Path
from rag_store import store_documents_batch, collection

KB_PATH = Path(__file__).resolve().parent.parent.parent / "jusbid_mongodb_fake_data" / "jusbid_mongodb_fake_data" / "knowledge_base.json"
POLICIES_PATH = Path(__file__).resolve().parent.parent.parent / "jusbid_mongodb_fake_data" / "jusbid_mongodb_fake_data" / "ai_policies.json"

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "business-hours-v1",
        "title": "Business Hours",
        "content": "Our demo business operates Monday through Saturday, from 9 AM to 6 PM. The business is closed on Sundays.",
        "keywords": ["hours", "schedule", "open"],
        "category": "operations",
    },
    {
        "doc_id": "refund-policy-v1",
        "title": "Refund Policy",
        "content": "Customers may submit refund requests within seven days of purchase, subject to the applicable terms and conditions.",
        "keywords": ["refund", "return", "money back"],
        "category": "payments",
    },
    {
        "doc_id": "human-support-v1",
        "title": "Human Support",
        "content": "If the approved knowledge base does not answer a question, the AI assistant should refer the customer to a human support representative.",
        "keywords": ["human", "support", "agent", "escalate"],
        "category": "support",
    },
]


def main():
    documents = []

    if KB_PATH.exists():
        print(f"Loading knowledge documents from {KB_PATH}...")
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_docs = json.load(f)
            print(f"Loaded {len(kb_docs)} knowledge base records.")
            documents.extend(kb_docs)
    else:
        print("knowledge_base.json not found, using fallback sample documents...")
        documents.extend(SAMPLE_DOCUMENTS)

    if POLICIES_PATH.exists():
        print(f"Loading AI policies from {POLICIES_PATH}...")
        with open(POLICIES_PATH, "r", encoding="utf-8") as f:
            policies = json.load(f)
            for pol in policies:
                documents.append({
                    "doc_id": pol.get("policy_id"),
                    "title": f"AI Policy: {pol.get('name')}",
                    "content": f"Policy rule: {pol.get('rule')}\nAction required: {pol.get('action')}",
                    "category": "policy",
                    "keywords": ["policy", "guideline", "rule", pol.get("name", "").lower(), pol.get("action", "").lower()],
                    "status": "approved",
                    "version": pol.get("version", 1),
                })
            print(f"Loaded {len(policies)} AI policy records.")

    print(f"\nEmbedding and indexing {len(documents)} total document(s) with Voyage AI ({collection.database.name}.{collection.name})...")

    # Clear stale records without embeddings before re-indexing
    collection.delete_many({"embedding": {"$exists": False}})

    count = store_documents_batch(documents=documents)
    print(f"\nSuccessfully indexed {count} document(s) into MongoDB '{collection.name}' collection!")
    print(f"Vector search index '{os.getenv('VECTOR_INDEX_NAME', 'knowledge_vector_index_jusbid')}' is active.")

    # Ingest any hotel records
    try:
        from ingest_hotels import ingest_hotels
        print("\nChecking and indexing hotel stays...")
        ingest_hotels()
    except Exception as e:
        print(f"Hotel indexing check: {e}")


if __name__ == "__main__":
    main()


