import re
import sys
from pathlib import Path
from bson import ObjectId

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from rag_store import collection, store_documents_batch, client


def ingest_hotels():
    print("Reading hotel records from MongoDB 'jusbid_ai.knowledge'...")

    # Find raw hotel documents
    raw_hotels = list(collection.find({"name": {"$exists": True}}))
    print(f"Found {len(raw_hotels)} raw hotel records.")

    seen_names = set()
    unique_hotels = []
    for h in raw_hotels:
        name = (h.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        unique_hotels.append(h)

    print(f"Identified {len(unique_hotels)} unique hotels to index into RAG.")

    documents_to_embed = []
    for h in unique_hotels:
        name = h.get("name", "").strip()
        price = h.get("price", "N/A")
        rating = h.get("rating", "N/A")
        reviews = h.get("reviews", "N/A")
        location = h.get("location", "N/A")
        contact = h.get("contact", "N/A")
        link = h.get("link", "")

        loc_text = location if location != "N/A" else "Udaipur, Rajasthan, India"
        price_text = f"{price} per night" if price != "N/A" else "Price available on application"
        rating_text = f"{rating} stars ({reviews} reviews)" if rating != "N/A" and reviews != "N/A" else "Ratings available on platform"
        contact_text = f"Phone / Contact: {contact}" if contact != "N/A" else "Contact via JusBid platform"

        text_content = (
            f"Hotel: {name}\n"
            f"Location: {loc_text}\n"
            f"Price: {price_text}\n"
            f"Rating: {rating_text}\n"
            f"{contact_text}\n"
            f"Booking & Details Link: {link}\n"
            f"About: Verified hotel stay in Udaipur, Rajasthan. Available for browsing and bidding on JusBid."
        )

        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:45]

        keywords = [
            "hotel", "stay", "udaipur", "room", "rooms", "resort", "homestay",
            "hostel", "accommodation", "booking", "udaipur hotel", name.lower()
        ]
        if location != "N/A":
            keywords.append(location.lower())

        documents_to_embed.append({
            "doc_id": f"HOTEL-{slug}",
            "source_id": f"HOTEL-{slug}",
            "title": f"Hotel: {name}",
            "category": "hotels",
            "content": text_content,
            "keywords": keywords,
            "status": "approved",
            "company_id": "jusbid",
            "version": 1,
            "hotel_details": {
                "name": name,
                "price": price,
                "rating": rating,
                "reviews": reviews,
                "location": location,
                "contact": contact,
                "link": link
            }
        })

    print(f"\nGenerating Voyage AI embeddings and indexing {len(documents_to_embed)} hotels...")
    count = store_documents_batch(documents_to_embed)
    print(f"Successfully vectorized and stored {count} hotel documents in 'jusbid_ai.knowledge'!")

    # Clean up the old raw documents with ObjectId keys that don't have embeddings
    del_res = collection.delete_many({
        "name": {"$exists": True},
        "_id": {"$type": "objectId"}
    })
    print(f"Cleaned up {del_res.deleted_count} raw unindexed hotel record(s).")
    print(f"Total documents now in collection: {collection.count_documents({})}")


if __name__ == "__main__":
    ingest_hotels()
