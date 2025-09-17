import os
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from .rag_dataset import rag_search, rag_stats

MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# In-memory session cache (simple for hackathon; swap for Firestore later if needed)
_LAST_LOCATION: Dict[str, Any] = {
    "input": "",
    "formatted": "",
    "lat": None,
    "lng": None,
    "types": [],  # e.g., ["locality", "political"] or ["administrative_area_level_1","political"]
}

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def _geocode(location: str) -> Dict[str, Any]:
    """Geocode free text into lat/lng + normalized address."""
    if not MAPS_KEY:
        # No live key -> fake success so the flow continues
        return {"ok": True, "formatted": location.strip(), "lat": None, "lng": None, "types": [], "note": "No GOOGLE_MAPS_API_KEY set; using fallback."}

    data = _http_get("https://maps.googleapis.com/maps/api/geocode/json", {"address": location, "key": MAPS_KEY})
    if data.get("_error"):
        return {"ok": False, "error": data["_error"]}
    if data.get("status") != "OK" or not data.get("results"):
        return {"ok": False, "error": f"Geocoding failed: {data.get('status')}"}

    r0 = data["results"][0]
    loc = r0["geometry"]["location"]
    return {"ok": True, "formatted": r0.get("formatted_address", location), "lat": loc.get("lat"), "lng": loc.get("lng"), "types": r0.get("types", [])}

def _parse_places(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        out.append({
            "name": r.get("name"),
            "address": r.get("formatted_address") or r.get("vicinity"),
            "rating": r.get("rating"),
            "open_now": (r.get("opening_hours") or {}).get("open_now"),
        })
    return out

def greeting() -> dict:
    """
    Dynamic first-turn greeting that reflects saved location and exposes
    all major capabilities as quick actions. Mentions dataset backing.
    """
    loc_status = get_saved_location()
    have_loc = loc_status.get("status") == "ok"
    loc_label = loc_status.get("formatted") if have_loc else None

    # dataset stats
    try:
        stats = rag_stats()  # builds index on first call
        kb_line = f"Scanning **~{stats['rows']:,} similar cases** from our library."
    except Exception:
        kb_line = "Scanning similar cases from our medical library."

    lines = []
    lines.append("**Hi, I’m CareGuide — your safety-first medical assistant.**")
    lines.append(kb_line)
    lines.append("**I can help right now:**")
    lines.append("- **Triage my symptoms** (rules + similar cases; not a diagnosis)")
    lines.append("- **Find nearby care**" + (f" (saved location: *{loc_label}*)" if have_loc else " (share your city/area)"))
    lines.append("- **Estimate typical costs** (clinic vs urgent care)")
    lines.append("- **What-If safety check** (e.g., “what if fever hits 39 °C?”)")
    lines.append("- **Medication side-effect check** (list your meds)")
    lines.append("- **Book an appointment** (simple confirmation)")
    lines.append("- **Fill my intake form** (with your consent)")
    lines.append("- **Save to my private timeline** (symptoms & actions)")
    if have_loc:
        lines.append(f"\nI’ve got you in **{loc_label}**. Want me to pull nearby options?")
    else:
        lines.append("\n**Before we start, what city/area are you in?**")
    lines.append("\n_I’ll include an **Evidence Panel** under answers so you can see sources & tools used._")
    lines.append("_This is general guidance, not a medical diagnosis._")

    actions = [
        {"id":"triage","label":"Triage my symptoms"},
        {"id":"nearby","label":"Find nearby care"},
        {"id":"costs","label":"Estimate costs"},
        {"id":"whatif","label":"What-If check"},
        {"id":"medsx","label":"Check med side-effects"},
        {"id":"book","label":"Book appointment"},
        {"id":"form","label":"Fill intake form"}
    ]
    _EVIDENCE.add("greeting","v3 menu rendered (dataset-backed)")
    return {"text":"\n".join(lines), "actions":actions, "have_location":have_loc, "location":loc_label}

def rag_search_tool(query: str, top_k: int = 3) -> list[dict]:
    """
    Wrapper so the agent can call RAG directly from the LLM plan.
    """
    results = rag_search(query, top_k=top_k)
    _EVIDENCE.add("RAG", f"{len(results)} results from dataset")
    return results


def set_user_location(location: str) -> dict:
    """Normalize and save the user's location."""
    if not location or not location.strip():
        return {"status": "error", "message": "Please share your city/area (you can include country)."}
    g = _geocode(location.strip())
    if not g.get("ok"):
        return {"status": "error", "message": f"Couldn't understand that location. {g.get('error','')}".strip()}
    _LAST_LOCATION.update({"input": location.strip(), "formatted": g["formatted"], "lat": g.get("lat"), "lng": g.get("lng"), "types": g.get("types", [])})
    return {"status": "ok", "saved_location": _LAST_LOCATION["formatted"], "note": g.get("note", "")}

def get_saved_location() -> dict:
    if not _LAST_LOCATION.get("formatted"):
        return {"status": "none", "message": "No saved location yet."}
    return {"status": "ok", **_LAST_LOCATION}

def _nearby_search(lat: float, lng: float, radius_m: int) -> List[Dict[str, Any]]:
    data = _http_get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        {"location": f"{lat},{lng}", "radius": radius_m, "keyword": "clinic hospital urgent care doctor", "key": MAPS_KEY},
    )
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"):
        return []
    return _parse_places(data)

def _text_search(query: str) -> List[Dict[str, Any]]:
    data = _http_get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        {"query": query, "key": MAPS_KEY},
    )
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"):
        return []
    return _parse_places(data)

def find_nearby_healthcare(location: Optional[str] = None, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Robust strategy:
      1) If a new `location` is provided, geocode + save it.
      2) If we have lat/lng -> try Nearby Search with expanding radii: 8km, 25km, 50km.
      3) If still empty, fall back to Text Search: "clinic OR urgent care OR hospital in {formatted}".
      4) If no API key, return static fallbacks.
    """
    # New location provided? Save it first.
    if location and location.strip():
        set_user_location(location)

    loc = _LAST_LOCATION.get("formatted")
    lat = _LAST_LOCATION.get("lat")
    lng = _LAST_LOCATION.get("lng")

    # No API key -> static mock so UX doesn't stall
    if not MAPS_KEY:
        place = loc or (location or "your area")
        return [
            {"name": "Nearby Clinic A", "address": place, "note": "Static fallback (set GOOGLE_MAPS_API_KEY for live results)"},
            {"name": "Nearby Clinic B", "address": place},
            {"name": "Urgent Care Center", "address": place},
        ][:max_results]

    # If we have lat/lng, Nearby Search first with growing radius
    results: List[Dict[str, Any]] = []
    if lat and lng:
        for r_m in (8000, 25000, 50000):  # 8 km, 25 km, 50 km
            results = _nearby_search(lat, lng, r_m)
            if results:
                break

    # Fallback: Text Search in the named place (works better for broad regions)
    if not results:
        q_place = loc or (location or "")
        results = _text_search(f"clinic OR urgent care OR hospital in {q_place}")

    if not results:
        hint = "Try a specific city or a zip/postal code."
        if loc:
            hint += f" (e.g., a neighborhood within {loc})"
        return [{"note": f"No healthcare places returned. {hint}"}]

    return results[:max_results]

# --- Cost & booking (unchanged) ---
_COST_TABLE = {
    "clinic_visit": {"insured": "USD 20–80 copay", "self_pay": "USD 80–250"},
    "flu_test": {"insured": "USD 0–40", "self_pay": "USD 20–120"},
    "strep_test": {"insured": "USD 0–40", "self_pay": "USD 20–80"},
    "urgent_care": {"insured": "USD 50–150 copay", "self_pay": "USD 120–350"},
}

def estimate_cost(has_insurance: bool, suspected: str = "") -> dict:
    plan = "insured" if has_insurance else "self_pay"
    items = ["clinic_visit"]
    s = (suspected or "").lower()
    if "flu" in s: items.append("flu_test")
    if "strep" in s or "sore throat" in s: items.append("strep_test")
    est = [{"item": it.replace("_"," "), "typical": _COST_TABLE[it][plan]} for it in items if it in _COST_TABLE]
    venue = "urgent care" if ("severe" in s or "chest pain" in s) else "clinic"
    venue_hint = _COST_TABLE["urgent_care" if venue=="urgent care" else "clinic_visit"][plan]
    return {"insurance": plan, "suggested_venue": venue, "venue_typical": venue_hint, "items": est}

def book_appointment(clinic_name: str, datetime_iso: str, reason: str = "") -> str:
    try:
        _ = datetime.fromisoformat(datetime_iso)
    except Exception:
        return "Please provide a valid ISO date/time, e.g., 2025-09-16T15:30:00."
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    return f"Booked a tentative appointment with **{clinic_name}** on **{datetime_iso}** (ID: {appt_id})."
