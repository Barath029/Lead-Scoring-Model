from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from feature_engineering import build_features, prepare_scoring_data

ROOT = Path(__file__).parent
pipeline = joblib.load(ROOT / "artifacts" / "pipeline.pkl")
app = FastAPI(title="CRM Lead Scoring API", version="1.0.0")


class LeadRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    lead_created_date: str
    last_activity_date: str
    lead_source: str
    industry: str | None = None
    region: str
    company_size: str
    job_seniority: str
    sales_rep: str
    website_visits: float = 0
    pages_viewed: float = 0
    emails_opened: float = 0
    emails_clicked: float = 0
    content_downloads: float = 0
    demo_requested: int = 0
    time_on_site_minutes: float = 0
    calls_count: float = 0
    meetings_booked: float = 0
    response_time_hours: float = 24


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(lead: LeadRequest) -> dict[str, float | str]:
    raw = prepare_scoring_data(pd.DataFrame([lead.model_dump()]))
    probability = float(pipeline.predict_proba(build_features(raw))[:, 1][0])
    band = "Hot" if probability >= 0.80 else "Warm" if probability >= 0.50 else "Cold"
    action = "Immediate sales call" if band == "Hot" else "Follow-up campaign" if band == "Warm" else "Nurture campaign"
    return {"conversion_probability": round(probability, 4), "score": round(probability * 100), "score_band": band, "recommended_action": action}
