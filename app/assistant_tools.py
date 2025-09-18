# app/assistant_tools.py
from __future__ import annotations

import os, re, time, uuid, csv, json, math
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse

# ------------------------
# Optional project modules
# ------------------------
# Evidence logger (safe fallback if missing)
try:
    from .evidence import EVIDENCE  # must expose .add(source, detail) and .snapshot(clear: bool)
except Exception:  # minimal shim
    class _Ev:
        _buf: List[Dict[str, str]] = []
        def add(self, source: str, detail: str) -> None:
            self._buf.append({"source": source, "detail": detail})
        def snapshot(self, clear: bool = True) -> List[Dict[str, str]]:
            items = list(self._buf)
            if clear:
                self._buf.clear()
            return items
    EVIDENCE = _Ev()

# RAG dataset hooks (safe fallback if missing)
try:
    from .rag_dataset import rag_search, rag_stats
except Exception:
    def rag_search(q: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return []
    def rag_stats() -> Dict[str, int]:
        return {"rows": 0}

# Triage pipeline (used elsewhere, not called here)
try:
    from .triage import triage_pipeline  # noqa: F401
except Exception:
    pass

# Optional centralized config
try:
    from .config import (
        GOOGLE_MAPS_API_KEY as _CONF_MAPS_KEY,
        HTTP_TIMEOUT_SECS as _CONF_TIMEOUT,
        PHI_ZERO_RETENTION as _CONF_ZERO_RET,
        EVIDENCE_ALLOWED_SOURCES as _CONF_EVIDENCE_SOURCES,
    )
except Exception:
    _CONF_MAPS_KEY = None
    _CONF_TIMEOUT = None
    _CONF_ZERO_RET = None
    _CONF_EVIDENCE_SOURCES = None

# ------------------------
# Environment / defaults
# ------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", _CONF_MAPS_KEY or "")
HTTP_TIMEOUT_SECS = int(os.getenv("HTTP_TIMEOUT_SECS", str(_CONF_TIMEOUT or 10)))
PHI_ZERO_RETENTION = str(os.getenv("PHI_ZERO_RETENTION", _CONF_ZERO_RET or "")).lower() in ("1", "true", "yes")
# Evidence sources allowed to render in panel (others are hidden)
EVIDENCE_ALLOWED_SOURCES = set(
    (_CONF_EVIDENCE_SOURCES or ["dataset", "rules", "rag", "whatif_rules", "medsx_dataset", "medsx_rules"])
)

MAPS_KEY = GOOGLE_MAPS_API_KEY

# ------------------------
# Small helpers / caching
# ------------------------
class TTLCache:
    def __init__(self, ttl_sec: int = 600, max_items: int = 256):
        self.ttl = ttl_sec
        self.max_items = max_items
        self.store: Dict[str, Tuple[float, Any]] = {}

    def get(self, k: str) -> Any | None:
        self._evict()
        x = self.store.get(k)
        if not x:
            return None
        ts, v = x
        if time.time() - ts > self.ttl:
            self.store.pop(k, None)
            return None
        return v

    def set(self, k: str, v: Any) -> None:
        self._evict()
        self.store[k] = (time.time(), v)

    def _evict(self) -> None:
        if len(self.store) <= self.max_items:
            return
        for k in sorted(self.store.keys(), key=lambda x: self.store[x][0])[: len(self.store) - self.max_items]:
            self.store.pop(k, None)

_HTTP_CACHE = TTLCache(ttl_sec=600)
_PLACES_CACHE = TTLCache(ttl_sec=600)

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = f"GET|{url}|{sorted(params.items())}"
    cached = _HTTP_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import requests  # local import to avoid hard dep if not needed
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECS)
        r.raise_for_status()
        data = r.json()
        _HTTP_CACHE.set(key, data)
        return data
    except Exception as e:
        return {"_error": str(e)}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        d = urlparse(url).netloc
        return d.replace("www.", "")
    except Exception:
        return url

def _tel_url(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    num = re.sub(r"[^\d+]", "", phone)
    return f"tel:{num}" if num else None

# ------------------------
# Evidence Panel (filtered)
# ------------------------
def evidence_snapshot(clear: bool = True) -> dict:
    """
    Return an evidence panel (filtered) for triage / what-if / meds flows.
    """
    items = EVIDENCE.snapshot(clear=False)
    filtered = [i for i in items if (i.get("source") in EVIDENCE_ALLOWED_SOURCES)]
    out = {"items": filtered}
    if clear and filtered:
        EVIDENCE.snapshot(clear=True)
    return out

# ------------------------
# Greeting (no startup location prompt)
# ------------------------
def greeting() -> dict:
    try:
        stats = rag_stats()
        kb_line = f"Scanning ~{stats.get('rows', 0):,} similar cases from our library."
    except Exception:
        kb_line = "Scanning similar cases from our medical library."

    lines: List[str] = []
    lines.append("**Hi, I’m CareGuide — your safety-first medical assistant.**")
    lines.append(kb_line)
    lines.append("**What I can do (reply with a number):**")
    lines.append("1) Triage my symptoms (not a diagnosis)")
    lines.append("2) Find nearby care (share your city/area)")
    lines.append("3) Estimate typical costs")
    lines.append("4) What-If safety check")
    lines.append("5) Medication side-effect check")
    lines.append("6) Book an appointment")
    lines.append("7) Fill intake form")
    lines.append("0) Main menu")
    lines.append("\nThis is general guidance, not a medical diagnosis.\n")
    lines.append("Disclaimer: This is general guidance, not a medical diagnosis.")

    EVIDENCE.add("greeting", "menu v4 (numbered, no startup location prompt)")
    return {"text": "\n".join(lines)}

# ------------------------
# Location memory (in-session only)
# ------------------------
_LAST_LOCATION: Dict[str, Any] = {
    "input": "", "formatted": "", "lat": None, "lng": None, "types": []
}

def set_user_location(location: str) -> dict:
    if not location or not location.strip():
        return {"status": "error", "message": "Please share your city/area."}
    g = _geocode(location.strip())
    if not g.get("ok"):
        return {"status": "error", "message": f"Couldn't understand that location. {g.get('error','')}".strip()}
    _LAST_LOCATION.update({
        "input": location.strip(),
        "formatted": g["formatted"],
        "lat": g.get("lat"),
        "lng": g.get("lng"),
        "types": g.get("types", [])
    })
    return {"status": "ok", "saved_location": _LAST_LOCATION["formatted"], "note": g.get("note","")}

def get_saved_location() -> dict:
    if not _LAST_LOCATION.get("formatted"):
        return {"status": "none", "message": "No saved location yet."}
    return {"status": "ok", **_LAST_LOCATION}

# ------------------------
# Google Maps / Places
# ------------------------
def _geocode(location: str) -> Dict[str, Any]:
    if not MAPS_KEY:
        return {
            "ok": True, "formatted": location.strip(), "lat": None, "lng": None, "types": [],
            "note": "No GOOGLE_MAPS_API_KEY set; using fallback."
        }
    data = _http_get("https://maps.googleapis.com/maps/api/geocode/json", {"address": location, "key": MAPS_KEY})
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

def _nearby_search(lat: float, lng: float, radius_m: int) -> List[Dict[str, Any]]:
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "keyword": "clinic hospital urgent care doctor",
        "key": MAPS_KEY
    }
    data = _http_get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params)
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"):
        return []
    out: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        geo = (r.get("geometry") or {}).get("location") or {}
        out.append({
            "name": r.get("name"),
            "address": r.get("vicinity") or r.get("formatted_address"),
            "rating": r.get("rating"),
            "place_id": r.get("place_id"),
            "google_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}" if r.get("place_id") else None,
            "open_now": ((r.get("opening_hours") or {}).get("open_now")),
            "lat": geo.get("lat"),
            "lng": geo.get("lng"),
        })
    return out

def _text_search(query: str) -> List[Dict[str, Any]]:
    data = _http_get("https://maps.googleapis.com/maps/api/place/textsearch/json", {"query": query, "key": MAPS_KEY})
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"):
        return []
    out: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        geo = (r.get("geometry") or {}).get("location") or {}
        out.append({
            "name": r.get("name"),
            "address": r.get("formatted_address") or r.get("vicinity"),
            "rating": r.get("rating"),
            "place_id": r.get("place_id"),
            "google_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}" if r.get("place_id") else None,
            "open_now": ((r.get("opening_hours") or {}).get("open_now")),
            "lat": geo.get("lat"),
            "lng": geo.get("lng"),
        })
    return out

def _place_details(place_id: str) -> Dict[str, Any]:
    if not MAPS_KEY or not place_id:
        return {}
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,international_phone_number,website,url",
        "key": MAPS_KEY
    }
    data = _http_get("https://maps.googleapis.com/maps/api/place/details/json", params)
    if data.get("_error") or data.get("status") not in ("OK",):
        return {}
    r = (data.get("result") or {})
    phone = r.get("formatted_phone_number") or r.get("international_phone_number")
    return {"phone": phone, "website": r.get("website"), "google_url": r.get("url")}

def find_nearby_healthcare(location: str = "", max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts enriched with phone/website/maps links.
    """
    if location and location.strip():
        set_user_location(location)
    loc = _LAST_LOCATION.get("formatted")
    lat = _LAST_LOCATION.get("lat")
    lng = _LAST_LOCATION.get("lng")

    # No Maps key → static fallback
    if not MAPS_KEY:
        place = loc or (location or "your area")
        return [
            {"name": "Nearby Clinic A", "address": place},
            {"name": "Nearby Clinic B", "address": place},
            {"name": "Urgent Care Center", "address": place},
        ][:max_results]

    cache_key = f"nearby|{lat}|{lng}|{loc}|{max_results}"
    if lat and lng:
        cached = _PLACES_CACHE.get(cache_key)
        if cached:
            return cached

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
        return [{"note": "No healthcare places returned. Try a specific city or a zip/postal code."}]

    enriched: List[Dict[str, Any]] = []
    for r in results[:max_results]:
        det = _place_details(r.get("place_id") or "")
        phone = det.get("phone")
        google_url = det.get("google_url") or r.get("google_url")
        website = det.get("website")
        item = {
            "name": r.get("name"),
            "address": r.get("address"),
            "rating": r.get("rating"),
            "phone": phone,
            "tel_url": _tel_url(phone),
            "website": website,
            "website_domain": _domain(website),
            "google_url": google_url,
            "open_now": r.get("open_now"),
            "lat": r.get("lat"),
            "lng": r.get("lng"),
        }
        enriched.append(item)
    _PLACES_CACHE.set(cache_key, enriched)
    return enriched

# ------------------------
# Costs & Booking
# ------------------------
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
    est = [{"item": it.replace("_", " "), "typical": _COST_TABLE[it][plan]} for it in items if it in _COST_TABLE]
    venue = "urgent care" if ("severe" in s or "chest pain" in s) else "clinic"
    venue_hint = _COST_TABLE["urgent_care" if venue == "urgent care" else "clinic_visit"][plan]
    return {"insurance": plan, "suggested_venue": venue, "venue_typical": venue_hint, "items": est}

def book_appointment(clinic_name: str, datetime_iso: str, reason: str = "") -> str:
    try:
        _ = datetime.fromisoformat(datetime_iso)
    except Exception:
        return "Please provide a valid ISO date/time, e.g., 2025-09-16T15:30:00."
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    return f"Booked a tentative appointment with **{clinic_name}** on **{datetime_iso}** (ID: {appt_id})."

# ------------------------
# RAG wrapper (for Evidence panel)
# ------------------------
def rag_search_tool(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar cases from the dataset.
    """
    try:
        results = rag_search(query, top_k=top_k)
    except Exception as e:
        EVIDENCE.add("dataset", f"rag error: {e}")
        return []
    names = [r.get("condition", "") for r in results if r.get("condition")]
    if names:
        EVIDENCE.add("dataset", f"rag top_k={len(results)} → {', '.join(names[:3])}")
    return results

# ------------------------
# What-If safety check (simple, explainable)
# ------------------------
def what_if_check(question_text: str) -> dict:
    """
    Returns risk band + reasons + actions based on conservative rules.
    """
    t = _norm(question_text)
    risk = "low"
    reasons: List[str] = []
    actions: List[str] = ["Monitor symptoms", "Hydrate and rest", "Seek care if symptoms worsen"]

    # Very simple illustrative rules (add more as needed)
    if re.search(r"\b39(\.|,)?\s?c|102(\.|,)?\s?f|high fever\b", t):
        risk = "moderate"; reasons.append("High fever can signal infection.")
        actions.insert(0, "Consider clinic evaluation within 24–48h")
    if re.search(r"\bchest pain|severe trouble breathing|shortness of breath\b", t):
        risk = "high"; reasons.append("Potential cardiopulmonary emergency.")
        actions = ["Seek emergency care now", "Avoid exertion", "Do not delay"]

    EVIDENCE.add("whatif_rules", f"risk={risk}, reasons={'; '.join(reasons) or 'none'}")
    return {"risk": risk, "reasons": reasons, "actions": actions}

# ------------------------
# Medication side-effects (multi-drug) with schema-safe signature
# ------------------------
_MEDS_DB: Optional[Dict[str, Dict[str, Any]]] = None

def _load_meds_db() -> None:
    """Load meds CSV if provided; else use a tiny built-in map."""
    global _MEDS_DB
    if _MEDS_DB is not None:
        return
    _MEDS_DB = {}
    # Try local CSV path first
    csv_path = os.getenv("MEDS_DATA_CSV")  # e.g., /app/data/meds.csv
    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = _norm(row.get("name", ""))
                    if not name:
                        continue
                    common = [x.strip() for x in (row.get("common_side_effects", "")).split("|") if x.strip()]
                    serious = [x.strip() for x in (row.get("serious_side_effects", "")).split("|") if x.strip()]
                    inter = [x.strip() for x in (row.get("interactions", "")).split("|") if x.strip()]
                    _MEDS_DB[name] = {"common": common, "serious": serious, "interactions": inter, "source": "csv"}
        except Exception as e:
            EVIDENCE.add("medsx_dataset", f"csv error: {e}")

    if not _MEDS_DB:
        # Minimal built-in fallback
        _MEDS_DB = {
            "ibuprofen": {
                "common": ["nausea", "heartburn", "stomach upset"],
                "serious": ["GI bleeding", "kidney issues"],
                "interactions": ["anticoagulants"],
                "source": "builtin"
            },
            "paracetamol": {
                "common": ["nausea", "rash (rare)"],
                "serious": ["liver injury (overdose)"],
                "interactions": ["alcohol"],
                "source": "builtin"
            },
            "amoxicillin": {
                "common": ["nausea", "diarrhea", "rash"],
                "serious": ["allergic reaction"],
                "interactions": ["warfarin"],
                "source": "builtin"
            },
        }

def _normalize_meds_list(inp) -> List[str]:
    """
    Accepts list[str] or a comma/newline/semicolon separated string.
    Returns a de-duplicated, normalized list of medication names.
    """
    if isinstance(inp, list):
        parts = inp
    else:
        parts = re.split(r"[,\n;]+", str(inp or ""))
    meds = []
    for p in parts:
        n = _norm(p)
        if not n:
            continue
        # example canonicalization
        if n == "acetaminophen":
            n = "paracetamol"
        meds.append(n)
    # de-duplicate
    seen = set()
    out: List[str] = []
    for m in meds:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

def meds_side_effects_check(meds: List[str], file_text: str = "") -> dict:
    """
    Multi-med side-effect checker with simple interaction merging.

    Parameters
    ----------
    meds : List[str]             # e.g., ["ibuprofen", "amoxicillin"]
    file_text : str              # optional extra text (OCR'd Rx)
    """
    _load_meds_db()
    assert _MEDS_DB is not None

    all_names = _normalize_meds_list(meds)
    if file_text:
        all_names.extend(_normalize_meds_list(file_text))
        all_names = _normalize_meds_list(all_names)

    if not all_names:
        return {"status": "error", "message": "No medication names detected."}

    merged_common: List[str] = []
    merged_serious: List[str] = []
    merged_interactions: List[str] = []
    details: List[dict] = []

    for name in all_names:
        rec = _MEDS_DB.get(name)
        if not rec:
            details.append({"drug": name, "found": False})
            continue
        details.append({"drug": name, "found": True, "source": rec.get("source")})
        for x in rec.get("common", []):
            if x not in merged_common:
                merged_common.append(x)
        for x in rec.get("serious", []):
            if x not in merged_serious:
                merged_serious.append(x)
        for x in rec.get("interactions", []):
            if x not in merged_interactions:
                merged_interactions.append(x)

    EVIDENCE.add("medsx_dataset", f"meds={all_names}, sources=mixed")
    return {
        "medications": all_names,
        "common_side_effects": merged_common[:12],
        "serious_side_effects": merged_serious[:12],
        "possible_interactions": merged_interactions[:12],
        "details": details,
    }

def meds_side_effects_check_text(meds_text: str, file_text: str = "") -> dict:
    """
    Convenience wrapper if the model/UI prefers a single string like "ibuprofen, amoxicillin".
    """
    return meds_side_effects_check(_normalize_meds_list(meds_text), file_text=file_text)

# ------------------------
# Simple timeline (in-memory, non-PHI if configured)
# ------------------------
_TIMELINE: List[Dict[str, str]] = []

def save_timeline(kind: str, details: Dict[str, str] | None = None) -> dict:
    """
    Save a lightweight timeline event (string-only details).
    """
    if PHI_ZERO_RETENTION:
        return {"status": "disabled", "message": "Zero-retention is enabled; timeline is off."}
    d = details or {}
    # enforce string-only values for schema simplicity
    d = {str(k): str(v) for k, v in d.items()}
    evt = {
        "id": "TL-" + uuid.uuid4().hex[:8].upper(),
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "kind": str(kind),
        "details": json.dumps(d, ensure_ascii=False),
    }
    _TIMELINE.insert(0, evt)
    if len(_TIMELINE) > 100:
        _TIMELINE.pop()
    return {"status": "ok", "event": evt}

def timeline_list() -> List[Dict[str, str]]:
    if PHI_ZERO_RETENTION:
        return []
    return list(_TIMELINE)

def timeline_clear() -> dict:
    if PHI_ZERO_RETENTION:
        return {"status": "disabled", "message": "Zero-retention is enabled; nothing stored."}
    _TIMELINE.clear()
    return {"status": "ok"}

# ------------------------
# Visit-prep utilities
# ------------------------
def visit_prep_summary(
    symptoms: str = "",
    duration: str = "",
    severity: str = "",
    meds: List[str] | None = None,
    allergies: List[str] | None = None,
    denied_red_flags: List[str] | None = None
) -> dict:
    """
    Produce a concise, shareable visit-prep summary (no diagnosis).
    """
    out = {
        "summary": {
            "symptoms": symptoms,
            "duration": duration,
            "severity": severity,
            "medications": meds or [],
            "allergies": allergies or [],
            "watchouts_denied": denied_red_flags or [],
        },
        "notes": "No diagnosis provided. For emergencies, seek immediate care.",
    }
    return out
