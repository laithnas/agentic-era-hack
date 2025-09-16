import os, requests, json, uuid
from datetime import datetime

MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyAAeyNburN-mdbW4ePe2QU5Yv3yLWf2tzQ")

def set_user_location(location: str) -> str:
    """
    Store/normalize location in-session. The LLM will remember it in context.
    Return a short confirmation the model can show.
    """
    loc = (location or "").strip()
    if not loc:
        return "I didn't catch the location. Please share your city/area."
    return f"Saved location: {loc}"

def find_nearby_healthcare(location: str, max_results: int = 3) -> list[dict]:
    """
    Use Google Places Text Search to find nearby clinics/urgent care/hospitals.
    Requires GOOGLE_MAPS_API_KEY. Returns a short list for the LLM to present.
    """
    if not location:
        return [{"note": "Missing location. Ask the user for their city/area first."}]

    if not MAPS_KEY:
        # Fallback placeholders if the API key isn't set yet
        return [
            {"name": "Nearby Clinic A", "address": f"{location}", "note": "Set GOOGLE_MAPS_API_KEY to enable live search"},
            {"name": "Nearby Clinic B", "address": f"{location}"},
            {"name": "Nearby Urgent Care", "address": f"{location}"}
        ]

    q = f"clinic OR urgent care OR hospital near {location}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": q, "key": MAPS_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        return [{"error": f"Places API error: {e}"}]

    out = []
    for r in data.get("results", [])[:max_results]:
        out.append({
            "name": r.get("name"),
            "address": r.get("formatted_address"),
            "rating": r.get("rating"),
            "open_now": (r.get("opening_hours") or {}).get("open_now")
        })
    return out or [{"note": "No places found. Try a different location."}]

_COST_TABLE = {
    # extremely rough demo ranges; tune to your locale or real payor data
    "clinic_visit": {"insured": "USD 20–80 copay", "self_pay": "USD 80–250"},
    "flu_test": {"insured": "USD 0–40", "self_pay": "USD 20–120"},
    "strep_test": {"insured": "USD 0–40", "self_pay": "USD 20–80"},
    "urgent_care": {"insured": "USD 50–150 copay", "self_pay": "USD 120–350"},
}

def estimate_cost(has_insurance: bool, suspected: str = "") -> dict:
    """
    Return a friendly cost snapshot based on insurance status and suspected context.
    """
    plan = "insured" if has_insurance else "self_pay"
    items = ["clinic_visit"]
    suspected_l = (suspected or "").lower()
    if "flu" in suspected_l:
        items.append("flu_test")
    if "strep" in suspected_l or "sore throat" in suspected_l:
        items.append("strep_test")

    est = []
    for it in items:
        row = _COST_TABLE.get(it, {})
        if row:
            est.append({"item": it.replace("_"," "), "typical": row.get(plan)})

    # Suggest venue by severity
    venue = "urgent care" if ("severe" in suspected_l or "chest pain" in suspected_l) else "clinic"
    venue_hint = _COST_TABLE.get("urgent_care" if venue=="urgent care" else "clinic_visit", {}).get(plan)

    return {"insurance": plan, "suggested_venue": venue, "venue_typical": venue_hint, "items": est}

def book_appointment(clinic_name: str, datetime_iso: str, reason: str = "") -> str:
    """
    Stub booking: just returns a confirmation code. In production, integrate the provider API.
    """
    try:
        _ = datetime.fromisoformat(datetime_iso)
    except Exception:
        return "Please provide a valid ISO date/time, e.g., 2025-09-16T15:30:00."
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    return f"Booked a tentative appointment with **{clinic_name}** on **{datetime_iso}** (ID: {appt_id})."
