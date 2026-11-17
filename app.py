import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from bson import ObjectId
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
bookings_collection = None


def init_mongo() -> bool:
    """Attempts to connect to MongoDB Atlas safely without throwing unhandled exceptions."""
    global mongo_client, mongo_collection, bookings_collection
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
        db_name = os.getenv("MONGODB_DATABASE", "jusbid_ai")
        mongo_collection = client[db_name][os.getenv("MONGODB_COLLECTION", "knowledge")]
        bookings_collection = client[db_name][os.getenv("MONGODB_BOOKINGS_COLLECTION", "bookings")]
        logger.info("MongoDB Atlas connected successfully (knowledge & bookings collections active).")
        return True
    except Exception as e:
        logger.warning(
            "MongoDB Atlas connection failed (%s). Operating safely in local knowledge cache mode.",
            e,
        )
        mongo_client = None
        mongo_collection = None
        bookings_collection = None
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
    customer_id: Optional[str] = None
    reservation_id: Optional[str] = None
    series: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


BOOKING_KEYWORDS = [
    "booking", "reservation", "check in", "check-in", "checkout", "check-out",
    "room number", "room no", "stay", "breakfast included", "meal plan",
    "property", "hotel", "my room", "booking status", "my reservation", "res-"
]


def is_booking_intent(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in BOOKING_KEYWORDS)


def extract_booking_identifiers(text: str) -> Dict[str, Optional[str]]:
    """Extracts possible reservation IDs, series codes, emails, or phone numbers from question text."""
    identifiers: Dict[str, Optional[str]] = {
        "reservation_id": None,
        "series": None,
        "email": None,
        "phone": None,
    }
    # Match RES-XXXXX-XXXXX...
    res_match = re.search(r'\b(RES-[A-Za-z0-9\-]+)\b', text, re.IGNORECASE)
    if res_match:
        identifiers["reservation_id"] = res_match.group(1).upper()

    # Match series BK-XXXXX
    series_match = re.search(r'\b(BK-[A-Za-z0-9]+)\b', text, re.IGNORECASE)
    if series_match:
        identifiers["series"] = series_match.group(1).upper()

    # Match email
    email_match = re.search(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', text)
    if email_match:
        identifiers["email"] = email_match.group(1).lower()

    # Match phone number (10 to 13 digits, optional +)
    phone_match = re.search(r'(\+?\d{10,13})', text)
    if phone_match:
        identifiers["phone"] = phone_match.group(1)

    return identifiers


def find_booking(request: ChatRequest) -> Optional[Dict[str, Any]]:
    """Queries MongoDB bookings collection using provided or extracted identifiers."""
    if bookings_collection is None:
        return None

    extracted = extract_booking_identifiers(request.question)
    res_id = request.reservation_id or extracted.get("reservation_id")
    series = request.series or extracted.get("series")
    email = request.email or extracted.get("email")
    phone = request.phone or extracted.get("phone")
    customer_id = request.customer_id

    query_parts: List[Dict[str, Any]] = []
    if res_id:
        query_parts.append({"reservation_id": {"$regex": f"^{re.escape(res_id)}$", "$options": "i"}})
    if series:
        query_parts.append({"series": {"$regex": f"^{re.escape(series)}$", "$options": "i"}})
    if email:
        query_parts.append({"guestEmail": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
    if phone:
        # Match by last 10 digits to be flexible with country code
        clean_phone = re.sub(r'\D', '', phone)
        suffix = clean_phone[-10:] if len(clean_phone) >= 10 else clean_phone
        query_parts.append({"guestContact": {"$regex": f"{re.escape(suffix)}"}})
    if customer_id:
        try:
            query_parts.append({"customerId": ObjectId(customer_id)})
        except Exception:
            query_parts.append({"customerId": customer_id})

    if not query_parts:
        return None

    try:
        booking = bookings_collection.find_one({"$or": query_parts})
        return booking
    except Exception as e:
        logger.warning("Error finding booking in MongoDB: %s", e)
        return None


def summarize_booking(doc: Dict[str, Any]) -> str:
    """Formats a clean, token-efficient summary of the MongoDB booking document."""
    prop_name = doc.get("property_name", "Hotel Property")
    res_id = doc.get("reservation_id", "N/A")
    series = doc.get("series", "N/A")
    guest_name = doc.get("guestName", "Guest")
    guest_email = doc.get("guestEmail", "N/A")
    guest_contact = doc.get("guestContact", "N/A")
    status = doc.get("status", "N/A")
    rate_plan = doc.get("rate_plan", "Standard Plan")
    payment_status = doc.get("paymentStatus", "N/A")
    total_amount = doc.get("totalAmount", doc.get("grandTotal", 0))
    total_paid = doc.get("totalPaid", 0)

    def _fmt_date(val: Any) -> str:
        if isinstance(val, dict) and "$date" in val:
            return str(val["$date"])[:10]
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d")
        return str(val)[:10] if val else "N/A"

    check_in = _fmt_date(doc.get("checkInDate"))
    check_out = _fmt_date(doc.get("checkOutDate"))

    # Room units
    room_units = doc.get("room_units", [])
    rooms_list = []
    for ru in room_units:
        if isinstance(ru, dict):
            r_no = ru.get("room_no", "Assigned upon check-in")
            r_type = ru.get("room_type", doc.get("room_name", "Room"))
            rooms_list.append(f"Room {r_no} ({r_type})")
    rooms_desc = ", ".join(rooms_list) if rooms_list else doc.get("room_name", "Standard Room")

    adults = doc.get("adultsCount", 1)
    children = doc.get("childrenCount", 0)

    # Payments info
    payments = doc.get("payments", [])
    tx_ids = [p.get("paymentTransactionId") for p in payments if isinstance(p, dict) and p.get("paymentTransactionId")]
    tx_str = ", ".join(tx_ids) if tx_ids else "N/A"

    return (
        f"Verified Reservation Details from Database:\n"
        f"• Property: {prop_name}\n"
        f"• Reservation ID: {res_id} (Booking Ref: {series})\n"
        f"• Guest Name: {guest_name} | Email: {guest_email} | Contact: {guest_contact}\n"
        f"• Reservation Status: {status}\n"
        f"• Check-in Date: {check_in}\n"
        f"• Check-out Date: {check_out}\n"
        f"• Assigned Room: {rooms_desc}\n"
        f"• Rate Plan & Meals: {rate_plan}\n"
        f"• Guests: {adults} Adult(s), {children} Child(ren)\n"
        f"• Payment Status: {payment_status} (Grand Total: {total_amount}, Total Paid: {total_paid})\n"
        f"• Transaction ID: {tx_str}\n"
    )


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


def normalize_greeting(text: str) -> str:
    """Normalizes greeting variations, punctuation, and collapsed letters (e.g., 'hiii' -> 'hi')."""
    clean = text.lower().strip(" !?.,;:-_~")
    # Collapse repeated characters (hiii -> hi, heyy -> hey, hellooo -> hello)
    clean = re.sub(r'h+i+', 'hi', clean)
    clean = re.sub(r'h+e+y+', 'hey', clean)
    clean = re.sub(r'h+e+l+o+', 'hello', clean)
    clean = re.sub(r'y+o+', 'yo', clean)
    return clean


GREETING_WORDS = {
    "hi", "hello", "hey", "hola", "namaste", "howdy", "yo", "sup",
    "greetings", "morning", "afternoon", "evening"
}

GREETING_PHRASES = {
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "how are you doing", "how are u", "how is it going",
    "hows it going", "how's it going", "what's up", "whats up",
    "who are you", "what can you do", "help me", "help",
    "thanks", "thank you", "thank u", "thx"
}


CONVERSATIONAL_PATTERNS = [
    r'\bhow are (you|u|you doing)\b',
    r'\bhow(\'s|s| is) it going\b',
    r'\bhow do you do\b',
    r'\bwhat(\'s|s| is) up\b',
    r'\bgood (morning|afternoon|evening|day|night)\b',
    r'\bwho are you\b',
    r'\bwhat can you do\b',
    r'\b(thank you|thanks|thank u|thx)\b',
    r'\bnice to meet you\b',
    r'\bhelp me\b',
]


def is_greeting(question: str) -> bool:
    norm = normalize_greeting(question)
    raw_clean = question.lower().strip(" !?.,;:-_~")

    # Match common conversational patterns (e.g. "how are you doing today", "good morning sir")
    if any(re.search(pat, raw_clean) for pat in CONVERSATIONAL_PATTERNS):
        return True

    words = norm.split()
    if not words:
        return False

    # Starts with a greeting word (e.g. "hi", "hii", "hello", "hey", "namaste") and is short
    if words[0] in GREETING_WORDS and len(words) <= 6:
        return True

    return False


def generate_greeting_response(question: str) -> Dict[str, Any]:
    """Generates a warm, natural, human-like concierge greeting."""
    system_prompt = (
        "You are the warm, polite, and attentive 24/7 AI Hotel & Support Concierge for JusBid.\n"
        "A guest has just greeted you or made friendly small talk.\n"
        "Guidelines:\n"
        "- Respond warmly, conversationally, and courteously, just like a professional 5-star hotel concierge.\n"
        "- Ask how you can assist them today with their hotel stay, reservation, or platform inquiries.\n"
        "- Keep your reply concise (1 to 2 sentences).\n"
        "- Sound natural and human. Do NOT dump bullet lists of rules or platform manuals unless specifically asked."
    )
    try:
        response = ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=80,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Greeting LLM error, using human fallback: %s", e)
        q = question.lower()
        if any(w in q for w in ["thank", "thx"]):
            answer = "You're very welcome! 😊 Please let me know if you need any other assistance. Have a wonderful day!"
        elif any(w in q for w in ["how are you", "how's it going", "hows it going"]):
            answer = "I'm doing well, thank you for asking! 😊 How can I help you with your hotel stay or booking today?"
        else:
            answer = "Hello! 👋 Welcome to JusBid. How can I assist you with your hotel reservation or stay today?"

    return {
        "answer": answer,
        "sources": [],
        "escalation_recommended": False,
    }


UNMATCHED_MESSAGE = (
    "Sorry, I can't help with that as it is outside our verified support policies.\n\n"
    "I can only help with:\n"
    "• **Your Hotel Bookings & Stays** (check-in/out dates, room numbers, meal plans, payment status)\n"
    "• **Bidding & Auction Rules** (reserve prices, bid increments, anti-sniping protection)\n"
    "• **Refund & Cancellation Policy** (eligibility, cancellation windows, deposit returns)\n"
    "• **Account Access & Security** (password reset, 2FA, identity verification)\n"
    "• **Payment Methods & Invoicing** (accepted cards, bank wire transfers, receipts)\n"
    "• **Fulfillment & Vouchers** (claiming certificates, redeeming won auctions)\n\n"
    "Please ask a question related to these topics, or click **'Quick Call'** to connect directly with a support specialist."
)


def generate_answer(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates customer support response using approved evidence."""
    # Friendly human greeting / small talk check
    if is_greeting(question):
        return generate_greeting_response(question)

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


def generate_booking_answer(question: str, booking: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates personalized concierge response using verified MongoDB booking details."""
    summary = summarize_booking(booking)
    policy_context = ""
    if evidence:
        policy_context = "\n\nOfficial Policy References:\n" + "\n\n".join(
            f"--- Policy: {item['title']} ---\n{item['text']}" for item in evidence
        )

    system_prompt = (
        "You are the official 24/7 AI Customer Support Concierge for JusBid.\n"
        "You are assisting a guest with inquiries regarding their verified hotel reservation.\n\n"
        "Guidelines:\n"
        "1. Use the Verified Reservation Details below to provide specific, polite, and accurate answers.\n"
        "2. Address the guest's question clearly: mention hotel property, check-in/out dates, room number, rate plan / meals, or payment details.\n"
        "3. If they ask about cancellation or refunds, evaluate their check-in date against the Official Policy References.\n"
        "4. Emphasize account security: remind guests never to share OTPs or passwords.\n"
        "5. Be concise, warm, professional, and use bullet points where helpful.\n"
        "6. Do not mention internal JSON keys or MongoDB; speak naturally as a hotel concierge."
    )

    user_prompt = (
        f"Guest Question: {question}\n\n"
        f"{summary}"
        f"{policy_context}\n\n"
        f"Please provide an accurate, friendly concierge response:"
    )

    try:
        response = ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Chat completion error on booking: %s", e)
        prop = booking.get('property_name', 'your hotel')
        res_id = booking.get('reservation_id', 'N/A')
        answer_text = (
            f"Here are the verified details for your reservation at **{prop}**:\n"
            f"• **Reservation ID**: {res_id}\n"
            f"• **Status**: {booking.get('status')}\n"
            f"• **Check-in**: {str(booking.get('checkInDate'))[:10]}\n"
            f"• **Check-out**: {str(booking.get('checkOutDate'))[:10]}\n"
            f"• **Rate Plan**: {booking.get('rate_plan')}\n"
            f"• **Payment Status**: {booking.get('paymentStatus')}\n\n"
            f"If you need further personal assistance with your stay, please click 'Quick Call' to speak directly with our concierge desk."
        )

    sources = [
        {
            "document_id": booking.get("reservation_id", "booking-record"),
            "title": f"Reservation: {booking.get('property_name', 'Hotel')}",
            "version": booking.get("series", "1.0"),
            "score": 1.0,
        }
    ]
    for ev in evidence:
        if ev.get("score", 0) >= 0.40:
            sources.append(ev)

    return {
        "answer": answer_text,
        "sources": sources,
        "escalation_recommended": False,
    }


@app.get("/health")
def health():
    db_status = "connected" if mongo_collection is not None else "local_cache_active"
    bookings_count = 0
    if bookings_collection is not None:
        try:
            bookings_count = bookings_collection.count_documents({})
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "JusBid AI Support Assistant",
        "database": db_status,
        "indexed_documents": len(memory_knowledge),
        "indexed_bookings": bookings_count,
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
        # 1. Check for booking inquiry or booking identifiers
        booking_intent = is_booking_intent(request.question)
        identifiers = extract_booking_identifiers(request.question)
        has_identifier = any([
            request.reservation_id, request.series, request.email, request.phone, request.customer_id,
            identifiers["reservation_id"], identifiers["series"], identifiers["email"], identifiers["phone"]
        ])

        if has_identifier or booking_intent:
            booking = find_booking(request)
            if booking:
                evidence = retrieve(request.question, limit=2)
                return generate_booking_answer(request.question, booking, evidence)
            elif has_identifier:
                return {
                    "answer": (
                        "I searched our reservation records, but could not find an active booking matching that ID, reference, or contact details. "
                        "Please verify your **Reservation ID** (e.g., `RES-...` or `BK-...`) or click **'Quick Call'** to connect with our front desk team."
                    ),
                    "sources": [],
                    "escalation_recommended": True,
                }
            elif booking_intent:
                return {
                    "answer": (
                        "I would be delighted to assist you with your reservation! 🏨\n\n"
                        "Please provide your **Reservation ID** (e.g. `RES-...` or `BK-...`) or your registered **Email / Phone Number** so I can pull up your stay details."
                    ),
                    "sources": [],
                    "escalation_recommended": False,
                }

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