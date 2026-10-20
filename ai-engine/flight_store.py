import os
import re
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://trail2000jan1_db_user:jkpvqFnD8gVChA4T@bot00.hmunzq1.mongodb.net/")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "jusbid_ai")
COLLECTION_NAME = os.getenv("FLIGHTS_COLLECTION", "flights_bom_blr")

_mongo_client = None


def get_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            MONGODB_URI,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
        )
    return _mongo_client[DATABASE_NAME][COLLECTION_NAME]


def is_flight_query(text: str) -> bool:
    """Detect whether user message is asking about flights, airlines, or airport routes."""
    t = text.lower()
    
    # Direct flight keywords
    if any(k in t for k in [
        "flight", "flights", "airfare", "air ticket", "airlines", "airline",
        "plane", "airport", "boarding", "takeoff", "landing"
    ]):
        return True
    
    # Route codes or city-to-city travel patterns
    if re.search(r"\b(bom|blr|mumbai|bangalore|bengaluru)\s*(to|->|-|se|ko|aur|and)\s*(bom|blr|mumbai|bangalore|bengaluru)\b", t):
        return True
    
    # Standalone mentions of flight codes or major Indian airlines with travel intent
    if re.search(r"\b(indigo|vistara|akasa|airasia)\b", t):
        return True
    if re.search(r"\bair\s*india\b", t):
        return True
    if re.search(r"\b(6e|uk|qp|i5)\s*\d{3,4}\b", t):
        return True
    if re.search(r"\bai\s*\d{3,4}\b", t):
        return True

    return False


def parse_flight_criteria(text: str):
    """Extract route, airline, timing, and sorting criteria from user text."""
    t = text.lower()
    
    origin = None
    destination = None
    airline = None
    sort_by = "price_asc"
    time_filter = None
    wants_all = False

    # Check for "all" / "every" / "complete"
    if any(w in t for w in ["all", "sari", "saari", "every", "complete list", "all the flight", "all flight", "pura", "pure"]):
        wants_all = True

    # Route detection
    bom_to_blr_patterns = [
        r"\b(bom|mumbai|bombay)\s*(to|->|-|se)\s*(blr|bangalore|bengaluru)\b",
        r"\b(from|se)\s*(bom|mumbai|bombay)\s*(to|ko)?\s*(blr|bangalore|bengaluru)\b",
        r"\bflight(s)?\s*(from\s+)?(bom|mumbai|bombay)\s+to\s+(blr|bangalore|bengaluru)\b",
        r"\bflight(s)?\s*(to|for)\s*(blr|bangalore|bengaluru)\b",
    ]
    blr_to_bom_patterns = [
        r"\b(blr|bangalore|bengaluru)\s*(to|->|-|se)\s*(bom|mumbai|bombay)\b",
        r"\b(from|se)\s*(blr|bangalore|bengaluru)\s*(to|ko)?\s*(bom|mumbai|bombay)\b",
        r"\bflight(s)?\s*(from\s+)?(blr|bangalore|bengaluru)\s+to\s+(bom|mumbai|bombay)\b",
        r"\bflight(s)?\s*(to|for)\s*(bom|mumbai|bombay)\b",
    ]

    for p in bom_to_blr_patterns:
        if re.search(p, t):
            origin = "Mumbai"
            destination = "Bengaluru"
            break

    if not origin:
        for p in blr_to_bom_patterns:
            if re.search(p, t):
                origin = "Bengaluru"
                destination = "Mumbai"
                break

    # If neither directional pattern matched, look for city mentions
    if not origin and not destination:
        has_bom = bool(re.search(r"\b(bom|mumbai|bombay)\b", t))
        has_blr = bool(re.search(r"\b(blr|bangalore|bengaluru)\b", t))
        
        if has_blr and not has_bom:
            if any(x in t for x in ["from bangalore", "from blr", "blr se", "bangalore se"]):
                origin = "Bengaluru"
                destination = "Mumbai"
            else:
                origin = "Mumbai"
                destination = "Bengaluru"
        elif has_bom and not has_blr:
            if any(x in t for x in ["to mumbai", "to bom", "mumbai ko", "bom ko"]):
                origin = "Bengaluru"
                destination = "Mumbai"
            else:
                origin = "Mumbai"
                destination = "Bengaluru"
        else:
            # Default to Mumbai -> Bengaluru (the primary sector)
            origin = "Mumbai"
            destination = "Bengaluru"

    # Airline filter with strict word boundaries
    if re.search(r"\bindigo\b|\b6e\b|\b6e\s*\d+", t):
        airline = "IndiGo"
    elif re.search(r"\bair\s*india\b|\bai\s*\d{3,4}\b", t):
        airline = "Air India"
    elif re.search(r"\bvistara\b|\buk\b|\buk\s*\d+", t):
        airline = "Vistara"
    elif re.search(r"\bakasa(\s*air)?\b|\bqp\b|\bqp\s*\d+", t):
        airline = "Akasa Air"
    elif re.search(r"\bair\s*asia\b|\bairasia\b|\bi5\b|\bi5\s*\d+", t):
        airline = "AirAsia"

    # Sorting
    if any(w in t for w in ["fastest", "quickest", "shortest", "least duration"]):
        sort_by = "duration_asc"
    elif any(w in t for w in ["earliest", "first", "early"]):
        sort_by = "departure_asc"
    elif any(w in t for w in ["latest", "last"]):
        sort_by = "departure_desc"
    elif any(w in t for w in ["expensive", "costliest", "highest price"]):
        sort_by = "price_desc"
    else:
        sort_by = "price_asc"  # cheapest first

    # Timing filter
    if any(w in t for w in ["morning", "subah", "early morning"]):
        time_filter = "morning"  # 05:00 - 11:59
    elif any(w in t for w in ["afternoon", "dopahar", "noon"]):
        time_filter = "afternoon"  # 12:00 - 16:59
    elif any(w in t for w in ["evening", "shaam"]):
        time_filter = "evening"  # 17:00 - 21:00
    elif any(w in t for w in ["night", "raat", "late night"]):
        time_filter = "night"  # 21:00 - 04:59

    return {
        "origin": origin,
        "destination": destination,
        "airline": airline,
        "sort_by": sort_by,
        "time_filter": time_filter,
        "wants_all": wants_all,
    }


def query_flights(query_text: str, max_results: int = 10):
    """Retrieve matching flights from MongoDB and generate structured context."""
    criteria = parse_flight_criteria(query_text)
    col = get_collection()

    mongo_filter = {}
    if criteria["origin"] and criteria["destination"]:
        mongo_filter["route"] = f"{criteria['origin']} -> {criteria['destination']}"
    elif criteria["origin"]:
        mongo_filter["departing_city"] = criteria["origin"]
    elif criteria["destination"]:
        mongo_filter["arriving_city"] = criteria["destination"]

    if criteria["airline"]:
        mongo_filter["airline"] = criteria["airline"]

    # Timing filter
    if criteria["time_filter"] == "morning":
        mongo_filter["departing_time"] = {"$gte": "05:00", "$lt": "12:00"}
    elif criteria["time_filter"] == "afternoon":
        mongo_filter["departing_time"] = {"$gte": "12:00", "$lt": "17:00"}
    elif criteria["time_filter"] == "evening":
        mongo_filter["departing_time"] = {"$gte": "17:00", "$lt": "21:30"}
    elif criteria["time_filter"] == "night":
        mongo_filter["$or"] = [
            {"departing_time": {"$gte": "21:30"}},
            {"departing_time": {"$lt": "05:00"}},
        ]

    # Sorting
    if criteria["sort_by"] == "price_asc":
        sort_clause = [("price_inr", 1), ("departing_time", 1)]
    elif criteria["sort_by"] == "price_desc":
        sort_clause = [("price_inr", -1)]
    elif criteria["sort_by"] == "duration_asc":
        sort_clause = [("duration_minutes", 1), ("price_inr", 1)]
    elif criteria["sort_by"] == "departure_asc":
        sort_clause = [("departing_time", 1)]
    elif criteria["sort_by"] == "departure_desc":
        sort_clause = [("departing_time", -1)]
    else:
        sort_clause = [("price_inr", 1)]

    # If the user asks for all flight data, return up to 50
    limit = 50 if criteria["wants_all"] else max_results
    cursor = col.find(mongo_filter, {"_id": 0}).sort(sort_clause).limit(limit)
    flights = list(cursor)

    # If filter with specific time yielded nothing, relax time filter
    if not flights and criteria["time_filter"]:
        mongo_filter.pop("departing_time", None)
        mongo_filter.pop("$or", None)
        cursor = col.find(mongo_filter, {"_id": 0}).sort(sort_clause).limit(limit)
        flights = list(cursor)

    route_name = f"{criteria['origin']} -> {criteria['destination']}" if criteria['origin'] else "Mumbai <-> Bengaluru"
    
    if flights:
        prices = [f["price_inr"] for f in flights]
        min_p = min(prices)
        max_p = max(prices)
        airlines_found = sorted(list(set(f["airline"] for f in flights)))
        total_in_db = col.count_documents({"route": route_name}) if criteria['origin'] else col.count_documents({})
        
        lines = [
            f"=== VERIFIED FLIGHT DATABASE RESULTS ({route_name}) ===",
            f"Total database flights on this route: {total_in_db}",
            f"Matching flights retrieved: {len(flights)}",
            f"Price range: ₹{min_p:,} to ₹{max_p:,}",
            f"Available airlines: {', '.join(airlines_found)}",
            "",
            "Flight Schedule & Fares:"
        ]
        
        for i, f in enumerate(flights, 1):
            next_day_str = " (+1 Day next morning)" if f.get("next_day") else ""
            lines.append(
                f"{i}. **{f['airline']}** ({f['flight_code']}): {f['departing_city']} [{f['departing_time']}] ➔ "
                f"{f['arriving_city']} [{f['arriving_time']}{next_day_str}] | "
                f"Duration: {f['duration']} | Fare: ₹{f['price_inr']:,}"
            )
        
        return "\n".join(lines), flights
    else:
        # Fallback to general route list
        all_cursor = col.find({"route": "Mumbai -> Bengaluru"}, {"_id": 0}).sort([("price_inr", 1)]).limit(8)
        fallback_flights = list(all_cursor)
        lines = [
            f"=== VERIFIED FLIGHT DATABASE RESULTS (Mumbai -> Bengaluru) ===",
            f"Showing top available flights:",
        ]
        for i, f in enumerate(fallback_flights, 1):
            next_day_str = " (+1 Day next morning)" if f.get("next_day") else ""
            lines.append(
                f"{i}. **{f['airline']}** ({f['flight_code']}): {f['departing_city']} [{f['departing_time']}] ➔ "
                f"{f['arriving_city']} [{f['arriving_time']}{next_day_str}] | "
                f"Duration: {f['duration']} | Fare: ₹{f['price_inr']:,}"
            )
        return "\n".join(lines), fallback_flights


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    test_queries = [
        "find flight from bom to blr",
        "make model capable to get all the flight data",
        "cheapest flight from bangalore to mumbai",
        "indigo flights from bom to blr",
        "morning flights from mumbai to bangalore",
    ]
    for q in test_queries:
        print("="*60)
        print(f"QUERY: {q}")
        print(f"Is flight query: {is_flight_query(q)}")
        context, flights = query_flights(q, max_results=5)
        print(context[:300] + "...")
