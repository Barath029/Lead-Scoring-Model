from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).parent / "data" / "leads_raw.csv"


def generate_dataset(n_leads: int = 10000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a reproducible, intentionally imperfect CRM lead dataset."""
    rng = np.random.default_rng(seed)
    created = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n_leads), unit="D"
    )
    source = rng.choice(
        ["organic", "paid_ads", "referral", "event", "cold_call", "webinar"],
        n_leads,
        p=[0.27, 0.20, 0.16, 0.12, 0.13, 0.12],
    )
    industry = rng.choice(
        ["SaaS", "Financial Services", "Healthcare", "Manufacturing", "Retail", "Education"],
        n_leads,
        p=[0.25, 0.18, 0.16, 0.16, 0.15, 0.10],
    )
    region = rng.choice(["North America", "EMEA", "APAC", "LATAM"], n_leads, p=[0.42, 0.28, 0.20, 0.10])
    company_size = rng.choice(["1-50", "51-200", "201-1000", "1001-5000", "5000+"], n_leads, p=[.22, .28, .25, .17, .08])
    seniority = rng.choice(["Individual Contributor", "Manager", "Director", "VP", "C-Level"], n_leads, p=[.30, .28, .22, .14, .06])
    lead_age = np.maximum(1, (pd.Timestamp("2026-01-01") - created).days)
    activity_days_ago = rng.integers(0, np.minimum(lead_age, 45) + 1)
    last_activity = pd.Series(created + pd.to_timedelta(activity_days_ago, unit="D"))
    last_activity = pd.to_datetime(last_activity)

    website_visits = rng.poisson(4 + 2 * (source == "organic") + 1 * (source == "webinar"))
    pages_viewed = np.maximum(1, website_visits + rng.poisson(3, n_leads))
    emails_opened = rng.poisson(3 + 1.5 * (source == "webinar"), n_leads)
    emails_clicked = np.minimum(emails_opened, rng.binomial(np.maximum(emails_opened, 1), 0.22))
    content_downloads = rng.poisson(1.2 + 0.5 * (industry == "SaaS"), n_leads)
    demo_requested = rng.binomial(1, np.clip(0.04 + 0.04 * (source == "referral") + 0.02 * (source == "event"), 0, .30))
    time_on_site = np.clip(rng.gamma(2.0, 2.5, n_leads) + pages_viewed * 0.8, 0.5, 90)
    calls = rng.poisson(1.2 + 0.8 * (source == "cold_call"), n_leads)
    meetings = rng.binomial(np.minimum(calls, 4), 0.22)
    response_hours = np.clip(rng.lognormal(2.2, 0.8, n_leads), 0.5, 240)
    reps = rng.choice(["Rep_A", "Rep_B", "Rep_C", "Rep_D", "Rep_E", "Rep_F"], n_leads)

    seniority_effect = pd.Series(seniority).map({"Individual Contributor": 0, "Manager": .15, "Director": .30, "VP": .42, "C-Level": .50}).to_numpy()
    source_effect = pd.Series(source).map({"organic": .15, "paid_ads": -.10, "referral": .55, "event": .20, "cold_call": -.20, "webinar": .28}).to_numpy()
    size_effect = pd.Series(company_size).map({"1-50": -.15, "51-200": .05, "201-1000": .20, "1001-5000": .30, "5000+": .24}).to_numpy()
    logit = (
        -3.15 + source_effect + size_effect + seniority_effect
        + 0.035 * website_visits + 0.025 * pages_viewed + 0.07 * emails_clicked
        + 0.18 * content_downloads + 0.95 * demo_requested + 0.10 * meetings
        - 0.004 * response_hours + 0.10 * (industry == "SaaS")
        + 0.16 * (np.isin(seniority, ["VP", "C-Level"]) & (demo_requested == 1))
        + rng.normal(0, 0.55, n_leads)
    )
    probability = 1 / (1 + np.exp(-logit))
    converted = rng.binomial(1, probability)
    conversion_delay = np.clip(rng.gamma(2.2, 18, n_leads).round().astype(int), 3, 180)
    conversion_date = created + pd.to_timedelta(conversion_delay, unit="D")
    conversion_date = conversion_date.where(converted == 1, pd.NaT)

    frame = pd.DataFrame({
        "lead_id": [f"L{number:06d}" for number in range(1, n_leads + 1)],
        "lead_created_date": created,
        "last_activity_date": last_activity,
        "lead_source": source,
        "industry": industry,
        "region": region,
        "company_size": company_size,
        "job_seniority": seniority,
        "sales_rep": reps,
        "website_visits": website_visits,
        "pages_viewed": pages_viewed,
        "emails_opened": emails_opened,
        "emails_clicked": emails_clicked,
        "content_downloads": content_downloads,
        "demo_requested": demo_requested,
        "time_on_site_minutes": time_on_site.round(2),
        "calls_count": calls,
        "meetings_booked": meetings,
        "response_time_hours": response_hours.round(2),
        "converted": converted,
        "conversion_date": conversion_date,
    })

    missing_columns = ["industry", "job_seniority", "response_time_hours", "content_downloads"]
    for column in missing_columns:
        mask = rng.random(n_leads) < 0.035
        frame.loc[mask, column] = np.nan
    outlier_mask = rng.random(n_leads) < 0.006
    frame.loc[outlier_mask, "website_visits"] *= 12
    return frame.sort_values("lead_created_date").reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(dataset):,} leads to {OUTPUT_PATH}")
    print(f"Conversion rate: {dataset['converted'].mean():.1%}")


if __name__ == "__main__":
    main()
