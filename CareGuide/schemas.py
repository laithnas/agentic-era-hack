# app/schemas.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field

class Place(BaseModel):
    name: str
    address: Optional[str] = None
    rating: Optional[float] = None
    phone: Optional[str] = None
    tel_url: Optional[str] = None
    website: Optional[HttpUrl] = None
    google_url: Optional[HttpUrl] = None

class TimelineEvent(BaseModel):
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class EvidencePanel(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
