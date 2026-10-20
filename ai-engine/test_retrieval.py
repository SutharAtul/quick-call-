
from rag_store import retrieve

# Ready Questions (Quick suggestion chips like in chatbot widget)
READY_QUESTIONS = [
    {"label": "Create Account", "question": "How do I create an account or sign up on JusBid?"},
    {"label": "Forgot Password", "question": "I forgot my password. What should I do?"},
    {"label": "Post Requirement", "question": "How can I post a requirement?"},
    {"label": "Submit a Bid", "question": "How do I submit a bid?"},
    {"label": "Track Bid Status", "question": "Where can I check my bid status?"},
    {"label": "Payment & Refund", "question": "My payment failed but money was deducted. What should I do?"},
    {"label": "Platform Fees", "question": "What are JusBid's fees?"},
    {"label": "Cancel a Bid", "question": "Can you cancel my bid for me?"},
    {"label": "Security / OTP", "question": "I received a suspicious message asking for my OTP."},
    {"label": "Human Support", "question": "Can I speak to a human support representative?"},
]

# Select questions to run (running first 3 by default to respect rate limits)
questions_to_test = READY_QUESTIONS[:3]

for item in questions_to_test:
    question = item["question"]
    print(f"\n[Chip: {item['label']}] Question: {question}")

    results = retrieve(
        question=question,
        limit=4,
    )


    if not results:
        print("No relevant documents found.")
        continue

    for result in results:
        print(
            f"- {result['title']} "
            f"(score: {result.get('score', 0):.3f})"
        )
        print(f"  {result['text']}")