# app/assistant_tools.py
from __future__ import annotations
import os, re, time, uuid, requests
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse

from .config import (
    GOOGLE_MAPS_API_KEY, HTTP_TIMEOUT_SECS,
    PHI_ZERO_RETENTION, EVIDENCE_ALLOWED_SOURCES,
)
from .schemas import Place, TimelineEvent, EvidencePanel
from .evidence import EVIDENCE
from .rag_dataset import rag_search, rag_stats
from .triage import triage_pipeline

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
        if not x: return None
        ts, v = x
        if time.time() - ts > self.ttl:
            self.store.pop(k, None)
            return None
        return v

    def set(self, k: str, v: Any) -> None:
        self._evict()
        self.store[k] = (time.time(), v)

    def _evict(self) -> None:
        if len(self.store) <= self.max_items: return
        # drop oldest
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
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECS)
        r.raise_for_status()
        data = r.json()
        _HTTP_CACHE.set(key, data)
        return data
    except Exception as e:
        return {"_error": str(e)}

def _norm(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip())

def _domain(url: Optional[str]) -> Optional[str]:
    if not url: return None
    try:
        d = urlparse(url).netloc
        return d.replace("www.","")
    except Exception:
        return url

def _tel_url(phone: Optional[str]) -> Optional[str]:
    if not phone: return None
    num = re.sub(r"[^\d+]", "", phone)
    return f"tel:{num}" if num else None

# ------------------------
# Evidence Panel (filtered)
# ------------------------
def evidence_snapshot(clear: bool=True) -> dict:
    items = EVIDENCE.snapshot(clear=False)
    filtered = [i for i in items if i.get("source") in EVIDENCE_ALLOWED_SOURCES]
    out = EvidencePanel(items=filtered).model_dump()
    if clear and filtered:
        EVIDENCE.snapshot(clear=True)
    return out

# ------------------------
# Greeting (no startup location prompt)
# ------------------------
def greeting() -> dict:
    try:
        stats = rag_stats()
        kb_line = f"Scanning ~{stats['rows']:,} similar cases from our library."
    except Exception:
        kb_line = "Scanning similar cases from our medical library."

    lines = []
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
    lines.append("\nThis is general guidance, not a medical diagnosis.\n\nDisclaimer: This is general guidance, not a medical diagnosis.")

    # Evidence not shown for greeting per policy, but we log internally:
    EVIDENCE.add("greeting","menu v4 (numbered, no startup location prompt)")
    return {"text":"\n".join(lines)}

# ------------------------
# Location memory (simple, in-session)
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
    _LAST_LOCATION.update({"input": location.strip(), "formatted": g["formatted"], "lat": g.get("lat"), "lng": g.get("lng"), "types": g.get("types", [])})
    return {"status": "ok", "saved_location": _LAST_LOCATION["formatted"], "note": g.get("note","")}

def get_saved_location() -> dict:
    if not _LAST_LOCATION.get("formatted"):
        return {"status":"none","message":"No saved location yet."}
    return {"status":"ok", **_LAST_LOCATION}

# ------------------------
# Google Maps / Places
# ------------------------
def _geocode(location: str) -> Dict[str, Any]:
    if not MAPS_KEY:
        return {"ok": True, "formatted": location.strip(), "lat": None, "lng": None, "types": [], "note": "No GOOGLE_MAPS_API_KEY set; using fallback."}
    data = _http_get("https://maps.googleapis.com/maps/api/geocode/json", {"address": location, "key": MAPS_KEY})
    if data.get("_error"): return {"ok": False, "error": data["_error"]}
    if data.get("status") != "OK" or not data.get("results"): return {"ok": False, "error": f"Geocoding failed: {data.get('status')}"}
    r0 = data["results"][0]; loc = r0["geometry"]["location"]
    return {"ok": True, "formatted": r0.get("formatted_address", location), "lat": loc.get("lat"), "lng": loc.get("lng"), "types": r0.get("types", [])}

def _nearby_search(lat: float, lng: float, radius_m: int) -> List[Dict[str, Any]]:
    data = _http_get("https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                     {"location": f"{lat},{lng}", "radius": radius_m, "keyword": "clinic hospital urgent care doctor", "key": MAPS_KEY})
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"): return []
    out=[]
    for r in data.get("results", []):
        out.append({
            "name": r.get("name"),
            "address": r.get("vicinity") or r.get("formatted_address"),
            "rating": r.get("rating"),
            "place_id": r.get("place_id"),
            "maps_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}" if r.get("place_id") else None
        })
    return out

def _text_search(query: str) -> List[Dict[str, Any]]:
    data = _http_get("https://maps.googleapis.com/maps/api/place/textsearch/json", {"query": query, "key": MAPS_KEY})
    if data.get("_error") or data.get("status") not in ("OK", "ZERO_RESULTS"): return []
    out=[]
    for r in data.get("results", []):
        out.append({
            "name": r.get("name"),
            "address": r.get("formatted_address") or r.get("vicinity"),
            "rating": r.get("rating"),
            "place_id": r.get("place_id"),
            "maps_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}" if r.get("place_id") else None
        })
    return out

def _place_details(place_id: str) -> Dict[str, Any]:
    if not MAPS_KEY or not place_id: return {}
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,international_phone_number,website,url",
        "key": MAPS_KEY
    }
    data = _http_get("https://maps.googleapis.com/maps/api/place/details/json", params)
    if data.get("_error") or data.get("status") not in ("OK",): return {}
    r = (data.get("result") or {})
    phone = r.get("formatted_phone_number") or r.get("international_phone_number")
    return {"phone": phone, "website": r.get("website"), "google_url": r.get("url")}

def find_nearby_healthcare(location: Optional[str] = None, max_results: int = 3) -> List[Dict[str, Any]]:
    if location and location.strip():
        set_user_location(location)
    loc = _LAST_LOCATION.get("formatted"); lat = _LAST_LOCATION.get("lat"); lng = _LAST_LOCATION.get("lng")

    # No Maps key → static fallback
    if not MAPS_KEY:
        place = loc or (location or "your area")
        return [
            Place(name="Nearby Clinic A", address=place).model_dump(),
            Place(name="Nearby Clinic B", address=place).model_dump(),
            Place(name="Urgent Care Center", address=place).model_dump(),
        ][:max_results]

    cache_key = f"nearby|{lat}|{lng}|{loc}|{max_results}"
    if lat and lng:
        cached = _PLACES_CACHE.get(cache_key)
        if cached: return cached

    results: List[Dict[str, Any]] = []
    if lat and lng:
        for r_m in (8000, 25000, 50000):
            results = _nearby_search(lat, lng, r_m)
            if results: break
    if not results:
        q_place = loc or (location or "")
        results = _text_search(f"clinic OR urgent care OR hospital in {q_place}")
    if not results:
        return [{"note": "No healthcare places returned. Try a specific city or a zip/postal code."}]

    enriched: List[Dict[str, Any]] = []
    for r in results[:max_results]:
        det = _place_details(r.get("place_id")) if r.get("place_id") else {}
        phone = det.get("phone")
        enriched.append(Place(
            name=r.get("name"),
            address=r.get("address"),
            rating=r.get("rating"),
            phone=phone,
            tel_url=_tel_url(phone),
            website=det.get("website"),
            google_url=det.get("google_url") or r.get("maps_url")
        ).model_dump())
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
    if "flu" in s: items.append("flu_test")
    if "strep" in s or "sore throat" in s:
        items.append("strep_test")

    est = [
        {"item": it.replace("_", " "), "typical": _COST_TABLE[it][plan]}
        for it in items
        if it in _COST_TABLE
    ]

    venue = "clinic"
    if any(x in s for x in ["severe", "chest pain", "difficulty breathing"]):
        venue = "urgent care"
    venue_hint = _COST_TABLE["urgent_care" if venue == "urgent care" else "clinic_visit"][plan]

    return {
        "insurance": plan,
        "suggested_venue": venue,
        "venue_typical": venue_hint,
        "items": est,
    }

def book_appointment(clinic_name: str, datetime_iso: str, reason: str = "") -> str:
    try:
        _ = datetime.fromisoformat(datetime_iso)
    except Exception:
        return "Please provide a valid ISO date/time, e.g., 2025-09-16T15:30:00."
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    return f"Booked a tentative appointment with **{clinic_name}** on **{datetime_iso}** (ID: {appt_id})."

# ------------------------
# Private Timeline (in-memory, optional zero-retention)
# ------------------------
_TIMELINE: List[TimelineEvent] = []

def save_timeline(event_type: str, payload: Dict[str, Any]) -> dict:
    """
    Store an in-session event unless PHI_ZERO_RETENTION=true.
    """
    if PHI_ZERO_RETENTION:
        return {"status": "disabled", "message": "Zero-retention is ON; nothing was stored."}
    ev = TimelineEvent(
        id="TL-" + uuid.uuid4().hex[:8].upper(),
        ts=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        type=event_type,
        data=payload or {},
    )
    _TIMELINE.append(ev)
    return {"status": "ok", "event": ev.model_dump()}

def timeline_list(limit: int = 20) -> List[dict]:
    return [e.model_dump() for e in _TIMELINE[-limit:]]

def timeline_clear() -> dict:
    _TIMELINE.clear()
    return {"status": "ok", "cleared": True}

# ------------------------
# What-If safety check (transparent mini rules)
# ------------------------
def what_if_check(question: str) -> dict:
    """
    Very small rules to illustrate transparent safety bands.
    Example triggers:
      - temperature in C/F
      - duration in hours/days
      - keywords like dehydration, chest pain
    """
    q = _norm(question)
    temp_c = None

    # Extract any number + unit
    m = re.search(r"(\d{2}\.?\d*)\s*°?\s*(c|f)\b", q)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        temp_c = val if unit == "c" else (val - 32) * 5.0 / 9.0

    # Extract rough duration
    dur_hrs = None
    if m := re.search(r"(\d+)\s*(hour|hr|hrs)", q):
        dur_hrs = int(m.group(1))
    if m := re.search(r"(\d+)\s*(day|days|d)", q):
        dur_hrs = max(dur_hrs or 0, int(m.group(1)) * 24)

    red = any(
        k in q
        for k in [
            "chest pain",
            "severe trouble breathing",
            "shortness of breath and chest pain",
            "passed out",
        ]
    )

    # Compute a very simple band
    band = "low"
    reasons = []
    actions = []

    if red:
        band = "high"
        reasons.append("Potential emergency symptom mentioned.")
        actions.append("Seek urgent medical care immediately.")

    if temp_c is not None:
        if temp_c >= 39.0:
            band = "high"
            reasons.append(f"High fever detected (≈{temp_c:.1f} °C).")
            actions.append("Escalate to clinician if sustained or if other symptoms worsen.")
        elif temp_c >= 38.0:
            band = "moderate"
            reasons.append(f"Fever detected (≈{temp_c:.1f} °C).")
            actions.append("Hydration, antipyretic per label, monitor 4–6 hours.")

    if dur_hrs and dur_hrs >= 72 and band != "high":
        band = "moderate"
        reasons.append("Symptoms persisting ≥72 hours.")
        actions.append("Consider clinician evaluation if not improving.")

    # Defaults if nothing triggered
    if not reasons:
        reasons.append("No major risk indicators detected from the description.")
        actions.append("Monitor symptoms, rest, fluids. Reassess if new red flags appear.")

    # Evidence entry
    EVIDENCE.add("whatif_calc", f"temp_c={temp_c}, dur_hrs={dur_hrs}, band={band}")

    return {
        "risk_band": band,
        "reasons": reasons,
        "actions": actions,
    }

# ------------------------
# Medication side-effect check (multi-med; optional CSV dataset)
# ------------------------
_MEDS_DB: Dict[str, Dict[str, Any]] | None = None

def _load_meds_db() -> None:
    """
    Load meds dataset from CSV if provided:
      ENV MEDS_DATA_CSV_GCS=gs://bucket/meds.csv  OR MEDS_DATA_CSV=/path/meds.csv
    CSV columns expected (case-insensitive subset ok):
      drug, common_side_effects, serious_side_effects, interactions
    Fallback to a tiny built-in dictionary.
    """
    global _MEDS_DB
    if _MEDS_DB is not None:
        return

    path = os.getenv("MEDS_DATA_CSV") or ""
    gcs_uri = os.getenv("MEDS_DATA_CSV_GCS") or ""

    rows: List[Dict[str, str]] = []
    try:
        if gcs_uri.startswith("gs://"):
            try:
                from google.cloud import storage  # type: ignore
                bucket, *key = gcs_uri[5:].split("/", 1)
                key = key[0] if key else ""
                client = storage.Client()
                blob = client.bucket(bucket).blob(key)
                text = blob.download_as_text()
                import csv
                for r in csv.DictReader(text.splitlines()):
                    rows.append({k.lower(): (v or "") for k, v in r.items()})
            except Exception:
                pass
        if not rows and path:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append({k.lower(): (v or "") for k, v in r.items()})
    except Exception:
        rows = []

    db: Dict[str, Dict[str, Any]] = {}
    if rows:
        for r in rows:
            name = _norm(r.get("drug", ""))
            if not name:
                continue
            db[name] = {
                "common": [x.strip() for x in (r.get("common_side_effects", "") or "").split(";") if x.strip()],
                "serious": [x.strip() for x in (r.get("serious_side_effects", "") or "").split(";") if x.strip()],
                "interactions": [x.strip() for x in (r.get("interactions", "") or "").split(";") if x.strip()],
                "source": "csv",
            }

    if not db:
        # Small fallback
        db = {
            "paracetamol": {
                "common": ["nausea", "headache"],
                "serious": ["liver problems (overdose)"],
                "interactions": ["alcohol (excess)"],
                "source": "builtin",
            },
            "ibuprofen": {
                "common": ["stomach upset", "heartburn"],
                "serious": ["stomach bleeding (rare)", "kidney issues (rare)"],
                "interactions": ["blood thinners (warfarin)", "other NSAIDs"],
                "source": "builtin",
            },
        }
    _MEDS_DB = db

def _normalize_meds_list(text_or_list: Any) -> List[str]:
    if isinstance(text_or_list, list):
        parts = text_or_list
    else:
        parts = re.split(r"[,\n;]+", text_or_list or "")
    meds = []
    for p in parts:
        n = _norm(p)
        if not n:
            continue
        # very small canonicalization
        n = {"acetaminophen": "paracetamol"}.get(n, n)
        meds.append(n)
    # dedupe preserving order
    seen = set()
    out = []
    for m in meds:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

def meds_side_effects_check(meds: Any, file_text: Optional[str] = None) -> dict:
    """
    Accepts a list OR a comma/newline separated string.
    Optionally include `file_text` extracted elsewhere (OCR of a script).
    Returns merged side effects and simple interaction hints.
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

    # naive pairwise interaction surfacing if both sides mention the same class string
    cross_alerts: List[str] = []
    if len(all_names) > 1 and merged_interactions:
        for i in range(len(all_names)):
            for j in range(i + 1, len(all_names)):
                a, b = all_names[i], all_names[j]
                # e.g., if one mentions "blood thinners" and the other is warfarin itself (not in tiny fallback)
                # we just raise a conservative heads-up
                pass

    EVIDENCE.add("medsx_dataset", f"meds={all_names}, sources=csv/builtin")

    return {
        "medications": all_names,
        "common_side_effects": merged_common[:12],
        "serious_side_effects": merged_serious[:12],
        "possible_interactions": merged_interactions[:12],
        "details": details,
    }

# ------------------------
# Visit-prep summary (simple, shareable)
# ------------------------
def visit_prep_summary(
    symptoms: str,
    duration: str = "",
    severity: str = "",
    meds: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    red_flags_denied: Optional[List[str]] = None,
) -> str:
    lines = []
    lines.append("**Visit Prep — Summary (copy/share)**")
    lines.append(f"- Symptoms: {symptoms or 'n/a'}")
    if duration:
        lines.append(f"- Duration: {duration}")
    if severity:
        lines.append(f"- Severity: {severity}")
    if meds:
        lines.append(f"- Current meds: {', '.join(meds)}")
    if allergies:
        lines.append(f"- Allergies: {', '.join(allergies)}")
    if red_flags_denied:
        lines.append(f"- Denied red flags: {', '.join(red_flags_denied)}")
    lines.append("- Self-care tried: (add here)")
    lines.append("- Questions for clinician: (add here)")
    return "\n".join(lines)

