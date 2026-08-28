from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "converted"
DATE_COLUMNS = ["lead_created_date", "last_activity_date"]
DROP_COLUMNS = ["lead_id", "conversion_date", TARGET]

NUMERIC_DEFAULTS = {
    "website_visits": 0,
    "pages_viewed": 0,
    "emails_opened": 0,
    "emails_clicked": 0,
    "content_downloads": 0,
    "demo_requested": 0,
    "time_on_site_minutes": 0,
    "calls_count": 0,
    "meetings_booked": 0,
    "response_time_hours": 24,
}
CATEGORICAL_DEFAULTS = {
    "lead_source": "unknown",
    "industry": "unknown",
    "region": "unknown",
    "company_size": "unknown",
    "job_seniority": "unknown",
    "sales_rep": "unassigned",
}
COLUMN_ALIASES = {
    "created_date": "lead_created_date",
    "lead_created": "lead_created_date",
    "created_at": "lead_created_date",
    "last_activity": "last_activity_date",
    "last_activity_at": "last_activity_date",
    "source": "lead_source",
    "leadsource": "lead_source",
    "company_industry": "industry",
    "seniority": "job_seniority",
    "job_title_seniority": "job_seniority",
    "website_visits_count": "website_visits",
    "web_visits": "website_visits",
    "demo_request": "demo_requested",
    "meetings": "meetings_booked",
}


def prepare_scoring_data(data: pd.DataFrame) -> pd.DataFrame:
    """Adapt partial CRM exports to the raw schema expected by the pipeline."""
    frame = data.copy()
    normalized = {column: str(column).strip().lower().replace(" ", "_") for column in frame.columns}
    frame = frame.rename(columns=normalized)
    frame = frame.rename(columns={column: COLUMN_ALIASES.get(column, column) for column in frame.columns})
    today = pd.Timestamp.today().normalize()
    if "lead_created_date" not in frame:
        frame["lead_created_date"] = today
    if "last_activity_date" not in frame:
        frame["last_activity_date"] = frame["lead_created_date"]
    for column, default in NUMERIC_DEFAULTS.items():
        if column not in frame:
            frame[column] = default
    for column, default in CATEGORICAL_DEFAULTS.items():
        if column not in frame:
            frame[column] = default
    return frame


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-aware features from raw CRM records."""
    frame = prepare_scoring_data(data)
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["lead_created_month"] = frame["lead_created_date"].dt.month.astype("float")
    frame["lead_created_quarter"] = frame["lead_created_date"].dt.quarter.astype("float")
    frame["lead_created_dow"] = frame["lead_created_date"].dt.dayofweek.astype("float")
    frame["lead_age_days"] = (frame["last_activity_date"] - frame["lead_created_date"]).dt.days.clip(lower=0)
    frame["recency_days"] = (pd.Timestamp("2026-01-01") - frame["last_activity_date"]).dt.days.clip(lower=0)
    frame["engagement_score"] = (
        frame["website_visits"].fillna(0) * 0.7
        + frame["pages_viewed"].fillna(0) * 0.35
        + frame["emails_opened"].fillna(0) * 0.4
        + frame["emails_clicked"].fillna(0) * 1.2
        + frame["content_downloads"].fillna(0) * 1.4
        + frame["demo_requested"].fillna(0) * 8
        + frame["meetings_booked"].fillna(0) * 3
    )
    frame["click_to_open_rate"] = frame["emails_clicked"] / frame["emails_opened"].replace(0, np.nan)
    frame["pages_per_visit"] = frame["pages_viewed"] / frame["website_visits"].replace(0, np.nan)
    frame["engagement_velocity"] = frame["engagement_score"] / (frame["lead_age_days"] + 1)
    frame["seniority_demo_interaction"] = frame["demo_requested"].fillna(0) * frame["job_seniority"].isin(["VP", "C-Level"]).astype(int)
    return frame.drop(columns=[column for column in DROP_COLUMNS if column in frame], errors="ignore")


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(data), data[TARGET].astype(int)
