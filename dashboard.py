from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

from feature_engineering import build_features, prepare_scoring_data

ROOT = Path(__file__).parent
pipeline = joblib.load(ROOT / "artifacts" / "pipeline.pkl")
st.set_page_config(page_title="Lead Scoring", page_icon="L", layout="wide")
st.title("CRM Lead Scoring")
st.caption("Prioritize current leads by predicted conversion likelihood within the modeled sales window.")

uploaded = st.file_uploader("Upload a CRM CSV", type="csv")
if uploaded is None:
    st.info("Upload a CSV with the same fields as data/leads_raw.csv to score a batch.")
    st.stop()

raw = pd.read_csv(uploaded)
raw = prepare_scoring_data(raw)
features = build_features(raw)
raw["conversion_probability"] = pipeline.predict_proba(features)[:, 1]
raw["score"] = (raw["conversion_probability"] * 100).round().astype(int)
raw["score_band"] = pd.cut(raw["conversion_probability"], [-1, .50, .80, 1], labels=["Cold", "Warm", "Hot"])
raw["recommended_action"] = raw["score_band"].map({"Hot": "Immediate sales call", "Warm": "Follow-up campaign", "Cold": "Nurture campaign"})
left, middle, right = st.columns([1, 1, 2])
with left:
    st.metric("Leads scored", f"{len(raw):,}")
    st.bar_chart(raw["score_band"].value_counts())
with middle:
    if "converted" in raw:
        st.metric("Converted leads", f"{int(raw['converted'].sum()):,}")
        st.metric("Conversion rate", f"{raw['converted'].mean():.1%}")
    st.metric("Hot leads", f"{int((raw['score_band'] == 'Hot').sum()):,}")
with right:
    st.subheader("Top leads")
    st.dataframe(raw.sort_values("conversion_probability", ascending=False).head(25), use_container_width=True)

st.download_button("Download scored CSV", raw.to_csv(index=False), "scored_leads.csv", "text/csv")
selected = st.selectbox("Inspect a lead", raw.index, format_func=lambda index: str(raw.loc[index].get("lead_id", index)))
if selected is not None:
    st.write({"lead_id": raw.loc[selected].get("lead_id", selected), "probability": round(float(raw.loc[selected, "conversion_probability"]), 4), "band": str(raw.loc[selected, "score_band"])})
    st.subheader("Why this score")
    transformed = pipeline.named_steps["features"].transform(features.loc[[selected]])
    estimator = pipeline.named_steps["model"]
    explanation = shap.LinearExplainer(estimator, transformed)(transformed)
    names = pipeline.named_steps["features"].get_feature_names_out()
    shap_values = explanation.values[0]
    explanation_table = pd.DataFrame({"feature": names, "impact": shap_values}).sort_values("impact", key=abs, ascending=False).head(10)
    st.bar_chart(explanation_table.set_index("feature"))
