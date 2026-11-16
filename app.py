import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from pymongo import MongoClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("jusbid-support")

# AI Client configuration
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = (
    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if os.getenv("OPENROUTER_API_KEY")
    else None
)

ai = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL") or os.getenv("OPENAI_CHAT_MODEL", "google/gemini-2.5-flash")
EMBED_MODEL = (
    os.getenv("OPENROUTER_EMBEDDING_MODEL")
    or os.getenv("OPENAI_EMBEDDING_MODEL")
    or "openai/text-embedding-3-small"
)
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))

# Database & Cache Setup
LOCAL_CACHE_PATH = Path("data/knowledge_cache.json")
FAQ_SOURCE_PATH = Path("data/faq.json")

# In-memory document store
memory_knowledge: List[Dict[str, Any]] = []
mongo_client: Optional[MongoClient] = None
mongo_collection = None


def init_mongo() -> bool:
    """Attempts to connect to MongoDB Atlas safely without throwing unhandled exceptions."""
    global mongo_client, mongo_collection
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        logger.info("MONGODB_URI not provided; running with local knowledge base.")
        return False
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=2000,
            tlsAllowInvalidCertificates=True,
        )
        client.admin.command("ping")
        mongo_client = client
        mongo_collection = client[
            os.getenv("MONGODB_DATABASE", "jusbid_ai")
        ][
            os.getenv("MONGODB_COLLECTION", "knowledge")
        ]
        logger.info("MongoDB Atlas connected successfully.")
        return True
    except Exception as e:
        logger.warning(
            "MongoDB Atlas connection failed (%s). Operating safely in local knowledge cache mode.",
            e,
        )
        mongo_client = None
        mongo_collection = None
        return False


def load_local_knowledge():
    """Loads knowledge cache from disk into memory."""
    global memory_knowledge
    if LOCAL_CACHE_PATH.exists():
        try:
            memory_knowledge = json.loads(LOCAL_CACHE_PATH.read_text(encoding="utf-8"))
            logger.info("Loaded %d knowledge records from local cache.", len(memory_knowledge))
            return
        except Exception as e:
            logger.error("Failed to parse %s: %s", LOCAL_CACHE_PATH, e)

    # Fallback to plain faq.json if cache not generated yet
    if FAQ_SOURCE_PATH.exists():
        try:
            docs = json.loads(FAQ_SOURCE_PATH.read_text(encoding="utf-8"))
            memory_knowledge = [
                {
                    "_id": f"{d['id']}-0",
                    "document_id": d["id"],
                    "title": d["title"],
                    "category": d.get("category", "general"),
                    "version": d.get("version", "1.0"),
                    "approved": d.get("approved", True),
                    "text": d["text"],
                    "embedding": None,
                }
                for d in docs
            ]
            logger.info("Loaded %d raw FAQ documents into memory fallback.", len(memory_knowledge))
        except Exception as e:
            logger.error("Failed to load %s: %s", FAQ_SOURCE_PATH, e)


# Initialize DB and Cache
init_mongo()
load_local_knowledge()

app = FastAPI(
    title="JusBid AI Customer Support Assistant",
    version="1.0.0",
    description="Intelligent 24/7 AI Customer Support Bot with RAG knowledge retrieval and human escalation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def embed(text: str) -> Optional[List[float]]:
    """Generates embedding vector for a given query text."""
    try:
        result = ai.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return result.data[0].embedding
    except Exception as e:
        logger.error("Failed to generate embedding for text: %s", e)
        return None


def retrieve_from_candidates(candidates: List[Dict[str, Any]], query_vector: Optional[List[float]], question: str, limit: int = 4):
    """Calculates cosine similarity and keyword relevance to return best matching evidence."""
    matches = []
    q_words = set(question.lower().split())

    q_norm = 0.0
    q_vec = None
    if query_vector:
        q_vec = np.array(query_vector, dtype=float)
        q_norm = float(np.linalg.norm(q_vec))

    for item in candidates:
        if not item.get("approved", True):
            continue

        vector = item.get("embedding")
        score = 0.0

        if q_vec is not None and vector and len(vector) == len(q_vec) and q_norm > 0:
            v_arr = np.array(vector, dtype=float)
            v_norm = float(np.linalg.norm(v_arr))
            if v_norm > 0:
                score = float(np.dot(q_vec, v_arr) / (q_norm * v_norm))
        else:
            # Keyword overlap fallback
            doc_words = set((item.get("title", "") + " " + item.get("text", "")).lower().split())
            overlap = len(q_words.intersection(doc_words))
            score = 0.2 + min(0.6, overlap * 0.1)

        matches.append({
            "title": item.get("title", "Support Knowledge"),
            "document_id": item.get("document_id", "doc"),
            "category": item.get("category", "support"),
            "version": item.get("version", "1.0"),
            "text": item.get("text", ""),
            "score": round(score, 4),
        })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return [m for m in matches[:limit] if m["score"] >= MIN_SIMILARITY]


def retrieve(question: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Retrieves relevant verified knowledge from MongoDB or the in-memory cache."""
    query_vector = embed(question)

    # 1. Try MongoDB if active
    if mongo_collection is not None:
        try:
            candidates = list(
                mongo_collection.find(
                    {"approved": True},
                    {
                        "title": 1,
                        "document_id": 1,
                        "category": 1,
                        "version": 1,
                        "text": 1,
                        "embedding": 1,
                    },
                )
            )
            if candidates:
                results = retrieve_from_candidates(candidates, query_vector, question, limit)
                if results:
                    return results
        except Exception as e:
            logger.warning("MongoDB query error: %s. Falling back to memory knowledge store.", e)

    # 2. Local memory knowledge store
    if memory_knowledge:
        return retrieve_from_candidates(memory_knowledge, query_vector, question, limit)

    return []


GREETINGS = {
    "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon",
    "good evening", "who are you", "what are you", "what can you do", "help",
    "how does this work", "how can you help", "how are you", "start"
}


def is_greeting(question: str) -> bool:
    clean = question.lower().strip(" !?.,;:")
    if clean in GREETINGS:
        return True
    if any(clean.startswith(g + " ") for g in ["hi", "hello", "hey", "good morning", "good evening"]):
        return len(clean.split()) <= 4
    return False


UNMATCHED_MESSAGE = (
    "Sorry, I can't help with that as it is outside our verified support policies.\n\n"
    "I can only help with:\n"
    "• **Bidding & Auction Rules** (reserve prices, bid increments, anti-sniping protection)\n"
    "• **Refund & Cancellation Policy** (eligibility, cancellation windows, deposit returns)\n"
    "• **Account Access & Security** (password reset, 2FA, identity verification)\n"
    "• **Payment Methods & Invoicing** (accepted cards, bank wire transfers, receipts)\n"
    "• **Fulfillment & Vouchers** (claiming certificates, redeeming won auctions)\n\n"
    "Please ask a question related to these topics, or click **'Quick Call'** to connect directly with a support specialist."
)


def generate_answer(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates customer support response using approved evidence."""
    # Friendly greeting / capability check
    if is_greeting(question):
        return {
            "answer": (
                "Hello! 👋 I'm your 24/7 JusBid AI Support Concierge.\n\n"
                "I can assist you with:\n"
                "• **Bidding & Auction Rules** (reserve prices, anti-sniping)\n"
                "• **Refunds & Cancellations** (timelines, deposit releases)\n"
                "• **Payments & Invoicing** (cards, wallets, bank wire)\n"
                "• **Account Access & 2FA** (password reset, identity verification)\n"
                "• **Voucher Claiming** (redeeming your won auctions)\n\n"
                "How may I assist you today? You can also click **'Quick Call'** anytime for immediate human concierge support."
            ),
            "sources": [],
            "escalation_recommended": False,
        }

    # If no evidence or top evidence score is too weak
    if not evidence or (evidence and evidence[0].get("score", 0) < 0.40):
        return {
            "answer": UNMATCHED_MESSAGE,
            "sources": [],
            "escalation_recommended": False,
        }

    # Format verified evidence context
    context = "\n\n".join(
        f"--- Source Document: {item['title']} (ID: {item['document_id']}, v{item['version']}) ---\n"
        f"{item['text']}"
        for item in evidence
    )

    system_prompt = (
        "You are the official 24/7 AI Customer Support Concierge for JusBid, a luxury auction and bidding platform.\n"
        "Your goal is to answer customer questions accurately, politely, and clearly based ONLY on the provided verified support knowledge excerpts.\n\n"
        "CRITICAL RULES:\n"
        "1. Check if the provided excerpts actually answer the customer's question.\n"
        "2. If the user's query is NOT addressed by the excerpts or asks for unrelated advice, DO NOT provide unrelated information from the excerpts. Instead, reply strictly with:\n"
        "'Sorry, I can't help with that as it is outside our verified support policies.\n\n"
        "I can only help with:\n"
        "• Bidding & Auction Rules (reserve prices, bid increments, anti-sniping protection)\n"
        "• Refund & Cancellation Policy (eligibility, cancellation windows, deposit returns)\n"
        "• Account Access & Security (password reset, 2FA, identity verification)\n"
        "• Payment Methods & Invoicing (accepted cards, bank wire transfers, receipts)\n"
        "• Fulfillment & Vouchers (claiming certificates, redeeming won auctions)\n\n"
        "Please ask a question related to these topics, or click Quick Call to speak directly with our concierge support team.'\n"
        "3. Never invent policies, deadlines, rates, or guarantees not stated in the sources.\n"
        "4. Emphasize account security: remind users never to share passwords or verification OTPs.\n"
        "5. Be concise, professional, warm, and helpful. Use clean bullet points where appropriate.\n"
        "6. Do NOT refer to 'the provided text' or 'the excerpts' in your answer; speak naturally as the company's customer support agent."
    )

    user_prompt = (
        f"Customer Question: {question}\n\n"
        f"Approved Knowledge:\n{context}\n\n"
        f"Please provide an accurate, friendly customer support response:"
    )

    try:
        response = ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Chat completion error: %s", e)
        # Fallback only if top evidence is a strong direct match
        top = evidence[0]
        if top.get("score", 0) >= 0.50:
            answer_text = (
                f"According to our official {top['title']}: {top['text']} "
                f"If you need further personal assistance, our concierge team is on standby."
            )
        else:
            return {
                "answer": UNMATCHED_MESSAGE,
                "sources": [],
                "escalation_recommended": False,
            }

    # If the answer indicates it cannot help, don't show irrelevant sources
    if "sorry, i can't help" in answer_text.lower():
        return {
            "answer": answer_text,
            "sources": [],
            "escalation_recommended": False,
        }

    return {
        "answer": answer_text,
        "sources": [
            {
                "document_id": item["document_id"],
                "title": item["title"],
                "version": item["version"],
                "score": item["score"],
            }
            for item in evidence
            if item.get("score", 0) >= 0.40
        ],
        "escalation_recommended": False,
    }


@app.get("/health")
def health():
    db_status = "connected" if mongo_collection is not None else "local_cache_active"
    return {
        "status": "ok",
        "service": "JusBid AI Support Assistant",
        "database": db_status,
        "indexed_documents": len(memory_knowledge),
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
    }


@app.get("/api/faq")
def get_faqs():
    """Returns FAQ list for front-end suggested questions and knowledge explorer."""
    if FAQ_SOURCE_PATH.exists():
        try:
            return json.loads(FAQ_SOURCE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return memory_knowledge


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        evidence = retrieve(request.question)
        return generate_answer(request.question, evidence)
    except Exception as e:
        logger.exception("Unexpected error handling chat request: %s", e)
        return {
            "answer": (
                "Our customer support assistant encountered a momentary hiccup. "
                "Please retry in a second or click 'Quick Call' to connect with a concierge specialist immediately."
            ),
            "sources": [],
            "escalation_recommended": True,
        }


# Serve frontend static assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "status": "ok",
        "service": "JusBid AI Support Prototype",
        "info": "Frontend files will be served once static/index.html is created.",
    }