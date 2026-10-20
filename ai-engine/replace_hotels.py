"""
replace_hotels.py
-----------------
Re-ingests hotel.json into the MongoDB knowledge collection and removes stale
hotel documents after the new records have been stored successfully.

Run from the ai-engine directory:
    python replace_hotels.py
"""

import json
import re
import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# hotel.json is two levels up from this script
HOTEL_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "hotel.json"

from rag_store import collection, store_documents_batch


def replace_hotels():
    print(f"Loading hotel data from: {HOTEL_JSON_PATH}")

    if not HOTEL_JSON_PATH.exists():
        print(f"ERROR: hotel.json not found at {HOTEL_JSON_PATH}")
        sys.exit(1)

    with open(HOTEL_JSON_PATH, "r", encoding="utf-8") as f:
        hotels = json.load(f)

    if not isinstance(hotels, list) or not hotels:
        raise ValueError("hotel.json must contain a non-empty list of hotel records.")

    print(f"Loaded {len(hotels)} hotels from hotel.json.")

    # Map and validate the complete input before making any MongoDB changes.
    documents_to_embed = []
    seen_names = set()

    for index, h in enumerate(hotels):
        if not isinstance(h, dict):
            raise ValueError(f"Hotel record {index + 1} must be a JSON object.")

        name_value = h.get("name")
        if not isinstance(name_value, str):
            raise ValueError(f"Hotel record {index + 1} must have a string name.")
        name = name_value.strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        hotel_id  = h.get("hotel_id", "")
        city      = h.get("city", "Unknown City")
        amenities = h.get("amenities", [])
        rate      = h.get("rate", {})
        avail     = h.get("availability", {})

        if not isinstance(amenities, list) or not all(
            isinstance(amenity, str) for amenity in amenities
        ):
            raise ValueError(f"Hotel '{name}' must have a list of string amenities.")
        if not isinstance(rate, dict) or not isinstance(avail, dict):
            raise ValueError(f"Hotel '{name}' has an invalid rate or availability object.")

        amount    = rate.get("amount", "N/A")
        currency  = rate.get("currency", "INR")
        per_night = "per night" if rate.get("per_night") else ""
        tax_note  = " (taxes included)" if rate.get("includes_taxes") else " (taxes extra)"
        price_text = (
            f"{currency} {amount} {per_night}{tax_note}".strip()
            if amount != "N/A"
            else "Price available on request"
        )

        amenities_text = ", ".join(amenities) if amenities else "Standard amenities"
        avail_status   = avail.get("status", "unknown")

        text_content = (
            f"Hotel: {name}\n"
            f"City / Location: {city}, Rajasthan, India\n"
            f"Price: {price_text}\n"
            f"Amenities: {amenities_text}\n"
            f"Availability: {avail_status}\n"
            f"About: Verified hotel available for browsing and bidding on JusBid."
        )

        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:45]

        keywords = [
            "hotel", "stay", "room", "rooms", "accommodation", "booking",
            "jusbid", name.lower(), city.lower(),
            f"{city.lower()} hotel",
        ] + [a.replace("_", " ") for a in amenities]

        documents_to_embed.append({
            "doc_id":     f"HOTEL-{slug}",
            "source_id":  f"HOTEL-{slug}",
            "title":      f"Hotel: {name}",
            "category":   "hotels",
            "content":    text_content,
            "keywords":   keywords,
            "status":     "approved",
            "company_id": "jusbid",
            "version":    1,
            "hotel_details": {
                "hotel_id":     hotel_id,
                "name":         name,
                "city":         city,
                "amenities":    amenities,
                "price":        price_text,
                "availability": avail_status,
            },
        })

    print(f"\nPrepared {len(documents_to_embed)} hotel documents for embedding.")
    if not documents_to_embed:
        raise ValueError("No valid hotel documents were found; MongoDB was not changed.")

    # Store first so embedding or insertion failures don't delete the old hotels.
    count = store_documents_batch(documents_to_embed)
    if count != len(documents_to_embed):
        raise RuntimeError(
            f"Stored {count} of {len(documents_to_embed)} hotel documents; "
            "stale hotel records were not removed."
        )

    expected_ids = [
        f"jusbid:{document['doc_id']}" for document in documents_to_embed
    ]
    del_result = collection.delete_many({
        "category": "hotels",
        "_id": {"$nin": expected_ids},
    })
    print(f"Removed {del_result.deleted_count} stale hotel document(s) from MongoDB.")

    # Remove only raw fixture records, not arbitrary knowledge documents with a hotel_id.
    del_raw = collection.delete_many({
        "hotel_id": {"$exists": True},
        "source": "prototype_fixture",
    })
    if del_raw.deleted_count:
        print(f"Removed {del_raw.deleted_count} raw fixture hotel record(s).")

    print(f"Successfully embedded and stored {count} hotel documents in MongoDB!")
    print(f"Total documents now in collection: {collection.count_documents({})}")


if __name__ == "__main__":
    replace_hotels()
