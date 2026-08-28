from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import split_features_target

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "leads_raw.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "pipeline.pkl"
METRICS_PATH = ROOT / "artifacts" / "model_comparison.csv"


def make_pipeline(model) -> Pipeline:
    categorical = ["lead_source", "industry", "region", "company_size", "job_seniority", "sales_rep"]
    numeric = [
        "website_visits", "pages_viewed", "emails_opened", "emails_clicked", "content_downloads",
        "demo_requested", "time_on_site_minutes", "calls_count", "meetings_booked", "response_time_hours",
        "lead_created_month", "lead_created_quarter", "lead_created_dow", "lead_age_days", "recency_days",
        "engagement_score", "click_to_open_rate", "pages_per_visit", "engagement_velocity", "seniority_demo_interaction",
    ]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("features", transformer), ("model", model)])


def main() -> None:
    raw = pd.read_csv(DATA_PATH, parse_dates=["lead_created_date", "last_activity_date", "conversion_date"])
    cutoff = raw["lead_created_date"].quantile(0.70)
    validation_cutoff = raw["lead_created_date"].quantile(0.85)
    train = raw[raw["lead_created_date"] < cutoff]
    validation = raw[(raw["lead_created_date"] >= cutoff) & (raw["lead_created_date"] < validation_cutoff)]
    test = raw[raw["lead_created_date"] >= validation_cutoff]
    x_train, y_train = split_features_target(train)
    x_validation, y_validation = split_features_target(validation)
    x_test, y_test = split_features_target(test)
    models = {
        "logistic_regression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
        "decision_tree": make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42)),
        "random_forest": make_pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    }
    results = []
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        results.append({"model": name, "validation_auc": roc_auc_score(y_validation, validation_probability), "validation_pr_auc": average_precision_score(y_validation, validation_probability)})
    best_name = max(results, key=lambda result: result["validation_pr_auc"])["model"]
    final_pipeline = models[best_name]
    final_pipeline.fit(pd.concat([x_train, x_validation]), pd.concat([y_train, y_validation]))
    test_probability = final_pipeline.predict_proba(x_test)[:, 1]
    results.append({"model": f"{best_name}_test", "validation_auc": roc_auc_score(y_test, test_probability), "validation_pr_auc": average_precision_score(y_test, test_probability)})
    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(final_pipeline, ARTIFACT_PATH)
    pd.DataFrame(results).to_csv(METRICS_PATH, index=False)
    (ROOT / "artifacts" / "training_summary.json").write_text(json.dumps({"best_model": best_name, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "test_auc": results[-1]["validation_auc"], "test_pr_auc": results[-1]["validation_pr_auc"]}, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
