# CRM Lead Scoring Model

A complete Business Analytics mini-project that estimates the probability a CRM lead converts within 90 days. The repository includes synthetic data generation, leakage-aware feature engineering, time-based evaluation, a FastAPI scoring endpoint, and a Streamlit business dashboard.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_generation.py
python train_model.py
python evaluate_model.py
```

The scripts create `data/leads_raw.csv`, `artifacts/pipeline.pkl`, model metrics, and evaluation plots. The generated data has a 13.5% conversion rate in the default seed and includes missing values, outliers, correlated engagement, and a conversion date.

## Run the applications

```powershell
uvicorn app:app --reload
streamlit run dashboard.py
```

The API is available at `http://127.0.0.1:8000`; use `GET /health` and `POST /predict`. The dashboard accepts a CSV containing the raw fields in `data/leads_raw.csv`, then provides ranked leads, score bands, and a downloadable scored file.

## Modeling choices

The default workflow uses a chronological 70% train, 15% validation, and 15% test split. Logistic regression, a pruned decision tree, and a balanced random forest are compared; validation PR-AUC selects the final pipeline because the positive class is relatively uncommon. Features are computed from information available at scoring time, and `conversion_date` is excluded to avoid leakage.

Score bands are Cold below 50%, Warm from 50% to 79%, and Hot at 80% or above. These thresholds should be tuned against capacity and conversion economics after CRM rollout.

## Docker and deployment

Build and run the API with:

```powershell
docker build -t lead-scoring .
docker run -p 8000:8000 lead-scoring
```

For Render or Railway, deploy this repository as a Docker web service and expose port 8000. For Streamlit Community Cloud, select `dashboard.py` as the entrypoint and ensure `artifacts/pipeline.pkl` is committed or generated during deployment.

## Data dictionary

`lead_id`, timestamps, categorical CRM descriptors, engagement counts (`website_visits`, `pages_viewed`, `emails_opened`, `emails_clicked`, `content_downloads`), intent (`demo_requested`), activity (`calls_count`, `meetings_booked`, `response_time_hours`), target `converted`, and target-only `conversion_date`.

## Monitoring

Monitor monthly score distributions, missingness, category novelty, and population stability by source and region. Compare predicted probabilities with actual 90-day outcomes, track calibration and top-quintile lift, and retrain quarterly or sooner when drift or campaign mix changes materially. Sales feedback should capture accepted, rejected, and converted leads so routing rules can be refined.
