# app/i18n.py
from __future__ import annotations

_LANG = "en"  # "en" | "es" | "fr" (extend)

def set_language(lang_code: str) -> dict:
    global _LANG
    lc = (lang_code or "").lower()
    if lc not in ("en","es","fr"):
        return {"status":"error","message":"Supported: en, es, fr"}
    _LANG = lc
    return {"status":"ok","lang":_LANG}

def get_language() -> dict:
    return {"lang": _LANG}

def phrase(key: str) -> dict:
    """
    Tiny dictionary for UX elements the model can choose to pull from.
    """
    T = {
        "en": {"menu_title":"What I can do (reply with a number):"},
        "es": {"menu_title":"¿Qué puedo hacer? (responde con un número):"},
        "fr": {"menu_title":"Ce que je peux faire (répondez par un numéro) :"},
    }
    return {"text": T.get(_LANG, T["en"]).get(key, key)}
