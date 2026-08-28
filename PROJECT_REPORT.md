# Lead Scoring Model: Project Report

## Executive summary

This project ranks CRM leads by the estimated probability of becoming a paying customer within 90 days. On realistic synthetic CRM data, the selected logistic-regression pipeline achieved a held-out test ROC-AUC of approximately 0.646 and PR-AUC of approximately 0.237. The model is intended to focus sales attention, not replace rep judgment.

## Business problem

Sales and marketing teams need a consistent way to prioritize follow-up. A conversion is defined as a lead progressing to a closed-won customer within 90 days of creation. Scores can support lead routing, SDR queue ordering, campaign audience selection, and capacity planning. Success should be measured with ROC-AUC, PR-AUC, precision@20%, lift, conversion-rate uplift, and rep time saved.

## Data and method

The synthetic dataset contains 10,000 leads with source, firmographics, seniority, engagement, CRM activity, timestamps, missing values, correlated variables, noise, and approximately 13.5% conversions. Features include recency, lead age, engagement score, engagement velocity, click-to-open rate, pages per visit, and a seniority/demo interaction. A chronological 70/15/15 split reflects deployment: older leads train the model and newer leads test it.

Class imbalance is handled first with balanced class weights. This preserves the original feature distribution and avoids synthetic records that may be unrealistic for a CRM. Logistic Regression, Decision Tree, and Random Forest are compared using validation PR-AUC; the operating threshold should be tuned with sales capacity and business cost data.

## Results and business use

The trained artifact is selected using validation PR-AUC and evaluated once on a future holdout. Run `python evaluate_model.py` to create ROC, precision-recall, calibration, and cumulative-gain charts in `artifacts/evaluation_plots.png`. A practical pilot would compare conversion rates for the top-scored quintile against the existing routing process, measure time-to-first-touch, and estimate incremental wins per rep-hour.

The API returns probability, a 0-100 score, Hot/Warm/Cold band, and a recommended action. Scores below 50 are Cold and assigned to nurture, 50-79 are Warm and assigned to follow-up, and 80-100 are Hot and assigned to immediate sales contact. The Streamlit dashboard supports batch upload, KPI cards, ranking, filtering through table sorting, SHAP explanations, and CSV download.

## Deployment and maintenance

Install requirements, generate data, train, evaluate, then run FastAPI or Streamlit as described in `README.md`. Package the API with the included Dockerfile. Monitor drift in score distributions, input missingness, new categories, calibration, and outcome rates by source and region. Retrain quarterly, or sooner when campaign mix, product, or routing policy changes.

## Limitations

This is synthetic data and the result is a proof of workflow, not a production performance claim. Real deployment requires a verified conversion definition, consent and governance review, historical snapshots that prevent future-activity leakage, and an experiment measuring incremental business impact.
