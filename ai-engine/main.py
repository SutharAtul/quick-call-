
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from rag_store import retrieve

app = FastAPI(title="AI Call Center Engine")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    status: str
    escalation_required: bool


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-engine"}


import re

HINGLISH_KEYWORDS = {
    "kaise", "kya", "kyu", "kyun", "kab", "kaha", "kahan", "hai", "hain", "kare",
    "karein", "karo", "karna", "karte", "karta", "karti", "mera", "meri", "mere",
    "mujhe", "mujhko", "apna", "apni", "apne", "aap", "tum", "tumhara", "tumhari",
    "nahi", "nahin", "batao", "bataye", "batayein", "bataiye", "samjhao", "gaya",
    "gayi", "gaye", "hoga", "hogi", "hoge", "hote", "hota", "hoti", "tha", "thi",
    "chahiye", "paise", "paisa", "rupaye", "bana", "banaye", "banayein",
    "bhej", "bhejo", "khol", "khole", "madad", "dikkat", "samasya", "bolo",
    "kripya", "theek", "thik", "wala", "wali", "wale", "kuch", "kuchh"
}


def detect_user_language(text: str) -> str:
    """Detect whether user's latest message is in Hindi/Hinglish or English."""
    # Check for Devanagari script
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"

    # Check for unambiguous Hinglish words
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if any(w in HINGLISH_KEYWORDS for w in words):
        return "hindi"

    return "english"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        raise HTTPException(
            status_code=503,
            detail="LLM configuration is missing"
        )

    # Contextual search query: for short follow-ups (e.g., "what is mean by route"), enrich with recent context
    raw_message = request.message.strip()
    search_query = raw_message
    if len(raw_message.split()) <= 6 and request.history:
        recent_text = " ".join(
            h.get("content", "") for h in request.history[-2:]
            if h.get("content")
        )
        search_query = f"{raw_message} {recent_text}"[:300]

    # Retrieve relevant knowledge from JusBid vector store / local fallback (retrieve 5 docs to give 3-4 choices)
    try:
        retrieved_docs = retrieve(
            question=search_query,
            limit=5,
        )
    except Exception as exc:
        print(f"Retrieval error: {exc}")
        retrieved_docs = []

    if retrieved_docs:
        knowledge_context = "\n\n".join(
            f"Document: {doc.get('title', 'Knowledge')}\n{doc.get('text', '')}"
            for doc in retrieved_docs
        )
    else:
        knowledge_context = (
            "No specific matching knowledge found in the approved documentation."
        )

    user_lang = detect_user_language(raw_message)

    system_prompt = """
You are a helpful and polite customer support assistant for JusBid.

Guidelines:
1. Base business facts, rules, and procedures strictly on the provided knowledge context.
2. Never invent unapproved policies, specific fee amounts, refund guarantees, account balances, or transaction statuses.
3. For conversational follow-ups and support term clarifications (such as asking what "route", "ticket", or a previous step means):
   - Explain clearly in plain, friendly language: "routing" means creating a support ticket or connecting the customer to a human support agent who can assist further.
   - Do NOT escalate simple language or term clarifications.
4. Only escalate (by beginning your response with "ESCALATE:") if:
   - The customer explicitly asks to speak to a person, supervisor, or human support agent.
   - The user requires private account changes, refunds, financial actions, or order status checks that cannot be completed by an automated assistant.
   - The user asks about a company fee or official policy that is absent from the knowledge base.
5. Language Handling (CRITICAL MANDATORY RULE):
   - Determine your response language SOLELY from the latest Customer message below (DO NOT carry over language from prior turns):
     * IF the latest customer message is in ENGLISH -> You MUST respond 100% in ENGLISH. (Even if previous turns in history were in Hindi, switch back to English immediately!)
     * IF the latest customer message is in HINGLISH or HINDI -> Respond in natural, friendly everyday Hindi (हिन्दी), keeping common technical, product, and UI keywords in English (e.g., Account, Register, Login, Password, OTP, Bid, Requirement, Payment, Refund, Status). Do not use archaic pure-Hindi words.
6. Recommendations & Options (Provide 3 to 4 Choices):
   - When the customer asks for hotel suggestions, options, or recommendations (e.g., in Udaipur or on JusBid), ALWAYS provide at least 3 to 4 of the best distinct choices from the provided knowledge context.
   - For each choice, format as a clean bullet point:
     * **Hotel Name**: Price per night, Star rating (review count), and Location / Area.
7. If the answer is supported or is a helpful explanation of support steps, respond directly with the answer.
8. Keep responses concise, warm, and natural for a live chat conversation.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation turns for context
    if request.history:
        for turn in request.history[-6:]:
            role = "assistant" if turn.get("role") in ("assistant", "bot") else "user"
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

    # Explicit instruction for this turn's response language
    lang_directive = (
        "Respond in ENGLISH."
        if user_lang == "english"
        else "Respond in conversational HINDI with English keywords."
    )

    # Latest customer message grounded with retrieved context
    user_prompt = (
        f"Business knowledge:\n{knowledge_context}\n\n"
        f"Language Directive: {lang_directive}\n"
        f"Customer message:\n{raw_message}"
    )
    messages.append({"role": "user", "content": user_prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 550,
    }


    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        answer = result["choices"][0]["message"]["content"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The model returned an empty response")

        answer = answer.strip()
        needs_human = answer.startswith("ESCALATE:")

        if needs_human:
            answer = answer.removeprefix("ESCALATE:").strip()

        return ChatResponse(
            answer=answer,
            status="escalation_required" if needs_human else "answered",
            escalation_required=needs_human,
        )


    except httpx.HTTPStatusError as exc:
        # Keep provider details and credentials out of customer responses.
        print(f"LLM provider returned HTTP {exc.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail="The AI provider request failed"
        ) from exc

    except (httpx.RequestError, ValueError, KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail="The AI service could not generate a response"
        )
