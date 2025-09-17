import os
import uuid
import urllib.parse
from typing import Optional, Dict, Any, List
from datetime import datetime

import requests

# ---- Evidence (real, in-memory) ----
# We keep a simple in-process log the agent can render in chat.
_EVIDENCE_LOG: List[Dict[str, str]] = []

class _Evidence:
    def add(self, source: str, detail: str):
        try:
            _EVIDENCE_LOG.append({"source": str(source), "detail": str(detail)})
        except Exception:
            pass

    def snapshot(self, clear: bool = False) -> List[Dict[str, str]]:
        items = list(_EVIDENCE_LOG)
        if clear:
            _EVIDENCE_LOG.clear()
        return items

_EVIDENCE = _Evidence()
# ------------------------------------

__all__ = [
    "greeting",
    "evidence_snapshot",      # NEW
    "triage_sources",         # NEW
    "rag_search_tool",
    "set_user_location",
    "get_saved_location",
    "find_nearby_healthcare",
    "estimate_cost",
    "book_appointment",
]

MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# In-memory session cache (simple for hackathon)
_LAST_LOCATION: Dict[str, Any] = {
    "input": "",
    "formatted": "",
    "lat": None,
    "lng": None,
    "types": [],
}

# ---------- HTTP helpers ----------
def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def _maps_link_from_place_id(place_id: Optional[str]) -> str:
    if not place_id:
        return "https://www.google.com/maps/search/?api=1&query=clinic"
    # Preferred pattern for place_id
    return f"https://www.google.com/maps/place/?q=place_id:{urllib.parse.quote(place_id)}"

# ---------- Geocode & Places ----------
def _geocode(location: str) -> Dict[str, Any]:
    """Geocode free text into lat/lng + normalized address."""
    if not MAPS_KEY:
        return {
            "ok": True,
            "formatted": location.strip(),
            "lat": None,
            "lng": None,
            "types": [],
            "note": "No GOOGLE_MAPS_API_KEY set; using fallback."
        }

    data = _http_get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        {"address": location, "key": MAPS_KEY},
    )
    if data.get("_error"):
        return {"ok": False, "error": data["_error"]}
    if data.get("status") != "OK" or not data.get("results"):
        return {"ok": False, "error": f"Geocoding failed: {data.get('status')}"}

    r0 = data["results"][0]
    loc = r0["geometry"]["location"]
    return {
        "ok": True,
        "formatted": r0.get("formatted_address", location),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "types": r0.get("types", []),
    }

def _parse_places(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        pid = r.get("place_id")
        name = r.get("name")
        addr = r.get("formatted_address") or r.get("vicinity")
        out.append({
            "name": name,
            "address": addr,
            "rating": r.get("rating"),
            "open_now": (r.get("opening_hours") or {}).get("open_now"),
            "maps_url": _maps_link_from_place_id(pid),   # clickable map
            "place_id": pid,
        })
    return out

def _place_details(place_id: str) -> Dict[str, Any]:
    """Fetch phone + website (+ Google URL) for a place_id."""
    if not (MAPS_KEY and place_id):
        return {}
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,international_phone_number,website,url",
        "key": MAPS_KEY,
    }
    data = _http_get("https://maps.googleapis.com/maps/api/place/details/json", params)
    if data.get("_error") or data.get("status") != "OK":
        return {}
    r = data.get("result", {})
    return {
        "phone": r.get("international_phone_number") or r.get("formatted_phone_number"),
        "website": r.get("website"),
        "google_url": r.get("url") or _maps_link_from_place_id(place_id),
    }

def _nearby_search(lat: float, lng: float, radius_m: int) -> List[Dict[str, Any]]:
    data = _http_get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        {
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "keyword": "clinic hospital urgent care doctor",
            "key": MAPS_KEY,
        },
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
def _tel_url(phone: Optional[str]) -> Optional[str]:
    """Create a tel: URL from a phone string."""
    if not phone:
        return None
    num = re.sub(r"[^\d+]", "", phone)
    return f"tel:{num}" if num else None

# ---------- Public tools ----------
def greeting() -> dict:
    """
    Dynamic first-turn greeting with numbered menu + 0) Main menu.
    """
    loc_status = get_saved_location()
    have_loc = loc_status.get("status") == "ok"
    loc_label = loc_status.get("formatted") if have_loc else None

    # Lazy import of RAG stats
    try:
        from .rag_dataset import rag_stats  # type: ignore
        stats = rag_stats()
        kb_line = f"Scanning ~{stats['rows']:,} similar cases from our library."
    except Exception:
        kb_line = "Scanning similar cases from our medical library."

    lines = []
    lines.append("**Hi, I’m CareGuide — your safety-first medical assistant.**")
    lines.append(kb_line)
    lines.append("**What I can do (reply with a number):**")
    lines.append("1) Triage my symptoms (not a diagnosis)")
    lines.append("2) Find nearby care" + (f" (saved location: *{loc_label}*)" if have_loc else " (share your city/area)"))
    lines.append("3) Estimate typical costs")
    lines.append("4) What-If safety check")
    lines.append("5) Medication side-effect check")
    lines.append("6) Book an appointment")
    lines.append("7) Fill intake form")
    lines.append("0) Main menu")
    if have_loc:
        lines.append(f"\nI’ve got you in **{loc_label}**. Want me to pull nearby options?")
    else:
        lines.append("\n**Before we start, what city/area are you in?**")
    lines.append("\nThis is general guidance, not a medical diagnosis.\n\nDisclaimer: This is general guidance, not a medical diagnosis.")

    actions = [
        {"id":"1","label":"Triage my symptoms"},
        {"id":"2","label":"Find nearby care"},
        {"id":"3","label":"Estimate costs"},
        {"id":"4","label":"What-If check"},
        {"id":"5","label":"Check med side-effects"},
        {"id":"6","label":"Book appointment"},
        {"id":"7","label":"Fill intake form"},
        {"id":"0","label":"Main menu"},
    ]
    _EVIDENCE.add("greeting","menu v4 (numbered, main menu option)")
    return {"text":"\n".join(lines), "actions":actions, "have_location":have_loc, "location":loc_label}

def evidence_snapshot(clear: bool = True) -> dict:
    """
    Return only evidence relevant to TRIAGE / MED SIDE-EFFECTS / WHAT-IF.
    We filter out greeting/places/cost/booking/etc.
    """
    items = _EVIDENCE.snapshot(clear=False)

    # Allow only these sources in the panel by default
    allowed = {
        # triage flow
        "dataset",          # from rag_search_tool()
        "triage_rules",     # if you add logging from rules engine later
        "triage_kb",        # if you log KB hits
        # meds side-effects flow (log with these source keys in your meds tool)
        "medsx_dataset",
        "medsx_rules",
        # what-if flow (log with these)
        "whatif_calc",
        "whatif_dataset",
    }

    filtered = [i for i in items if i.get("source") in allowed]
    # If nothing matched, return empty list but don't clear the underlying log yet
    out = {"items": filtered}

    # Clear the underlying log only if we emitted something for the user
    if clear and filtered:
        _EVIDENCE.snapshot(clear=True)
    return out

def triage_sources() -> dict:
    """
    Static sources for triage (so the agent can show them in Evidence Panel).
    """
    try:
        from .triage import DATA_PATH as TRIAGE_DATA_PATH, SYMPTOM_KB_PATH as TRIAGE_KB_PATH  # type: ignore
        return {
            "rules_path": str(TRIAGE_DATA_PATH),
            "kb_path": str(TRIAGE_KB_PATH),
        }
    except Exception:
        return {
            "rules_path": "app/data/conditions.json",
            "kb_path": "app/data/kb_symptom_to_conditions.json",
        }

def rag_search_tool(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """RAG over gs://lohealthcare/ai-medical-chatbot.csv (lazy import)."""
    try:
        from .rag_dataset import rag_search  # type: ignore
        results = rag_search(query, top_k=top_k)
    except Exception:
        results = []
    # Put a compact summary into evidence
    if results:
        summary = "; ".join([f"{i+1}) sim={r.get('similarity')} «{(r.get('description') or '')[:40]}…»" for i, r in enumerate(results[:3])])
        _EVIDENCE.add("dataset", f"{len(results)} similar cases: {summary}")
    else:
        _EVIDENCE.add("dataset", "no similar cases found")
    return results

def set_user_location(location: str) -> dict:
    """Normalize and save the user's location."""
    if not location or not location.strip():
        return {"status": "error", "message": "Please share your city/area (you can include country)."}
    g = _geocode(location.strip())
    if not g.get("ok"):
        return {"status": "error", "message": f"Couldn't understand that location. {g.get('error','')}".strip()}
    _LAST_LOCATION.update({
        "input": location.strip(),
        "formatted": g["formatted"],
        "lat": g.get("lat"),
        "lng": g.get("lng"),
        "types": g.get("types", []),
    })
    _EVIDENCE.add("geocode", _LAST_LOCATION["formatted"])
    return {"status": "ok", "saved_location": _LAST_LOCATION["formatted"], "note": g.get("note", "")}

def get_saved_location() -> dict:
    if not _LAST_LOCATION.get("formatted"):
        return {"status": "none", "message": "No saved location yet."}
    return {"status": "ok", **_LAST_LOCATION}

def find_nearby_healthcare(location: Optional[str] = None, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Strategy:
      1) If a new `location` is provided, geocode + save it.
      2) If we have lat/lng -> Nearby Search with 8/25/50km.
      3) Else Text Search in the named place.
      4) Enrich top results with phone + website (Place Details).
      5) If no API key, return static fallbacks with generic map link.
    """
    if location and location.strip():
        set_user_location(location)

    loc = _LAST_LOCATION.get("formatted")
    lat = _LAST_LOCATION.get("lat")
    lng = _LAST_LOCATION.get("lng")

    # No live key -> static mock
    if not MAPS_KEY:
        place = loc or (location or "your area")
        generic_url = "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote_plus(f"clinic in {place}")
        _EVIDENCE.add("places", "fallback/static (no API key)")
        return [
            {"name": "Nearby Clinic A", "address": place, "maps_url": generic_url, "note": "Static fallback — set GOOGLE_MAPS_API_KEY", "phone": None, "website": None},
            {"name": "Nearby Clinic B", "address": place, "maps_url": generic_url, "phone": None, "website": None},
            {"name": "Urgent Care Center", "address": place, "maps_url": generic_url, "phone": None, "website": None},
        ][:max_results]

    results: List[Dict[str, Any]] = []
    if lat and lng:
        for r_m in (8000, 25000, 50000):
            results = _nearby_search(lat, lng, r_m)
            if results:
                break

    if not results:
        q_place = loc or (location or "")
        results = _text_search(f"clinic OR urgent care OR hospital in {q_place}")

    if not results:
        hint = "Try a specific city or a zip/postal code."
        if loc:
            hint += f" (e.g., a neighborhood within {loc})"
        _EVIDENCE.add("places", "no results")
        return [{"note": f"No healthcare places returned. {hint}"}]

    # Enrich top-k with details
    # Enrich top-k with details
    enriched: List[Dict[str, Any]] = []
    for r in results[:max_results]:
        det = _place_details(r.get("place_id")) if r.get("place_id") else {}
        phone = det.get("phone")
        enriched.append({
            **r,
            "phone": phone,
            "tel_url": _tel_url(phone),                           # NEW
            "website": det.get("website"),
            "google_url": det.get("google_url") or r.get("maps_url"),
        })
    _EVIDENCE.add("places", f"{len(enriched)} results with details")
    return enriched

# ---------- Cost & booking ----------
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
    if "flu" in s:
        items.append("flu_test")
    if "strep" in s or "sore throat" in s:
        items.append("strep_test")
    est = [
        {"item": it.replace("_"," "), "typical": _COST_TABLE[it][plan]}
        for it in items if it in _COST_TABLE
    ]
    venue = "urgent care" if ("severe" in s or "chest pain" in s) else "clinic"
    venue_hint = _COST_TABLE["urgent_care" if venue == "urgent care" else "clinic_visit"][plan]
    _EVIDENCE.add("cost", f"{plan}, venue={venue}, items={','.join([e['item'] for e in est])}")
    return {"insurance": plan, "suggested_venue": venue, "venue_typical": venue_hint, "items": est}

def book_appointment(clinic_name: str, datetime_iso: str, reason: str = "") -> str:
    try:
        _ = datetime.fromisoformat(datetime_iso)
    except Exception:
        return "Please provide a valid ISO date/time, e.g., 2025-09-16T15:30:00."
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    _EVIDENCE.add("booking", f"{clinic_name} @ {datetime_iso}")
    return f"Booked a tentative appointment with **{clinic_name}** on **{datetime_iso}** (ID: {appt_id})."
