from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

from feature_engineering import split_features_target

ROOT = Path(__file__).parent


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "leads_raw.csv", parse_dates=["lead_created_date", "last_activity_date", "conversion_date"])
    cutoff = raw["lead_created_date"].quantile(0.85)
    test = raw[raw["lead_created_date"] >= cutoff]
    features, target = split_features_target(test)
    pipeline = joblib.load(ROOT / "artifacts" / "pipeline.pkl")
    probability = pipeline.predict_proba(features)[:, 1]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    fpr, tpr, _ = roc_curve(target, probability)
    axes[0, 0].plot(fpr, tpr, label=f"AUC {roc_auc_score(target, probability):.3f}"); axes[0, 0].plot([0, 1], [0, 1], "--"); axes[0, 0].legend(); axes[0, 0].set_title("ROC curve")
    precision, recall, _ = precision_recall_curve(target, probability)
    axes[0, 1].plot(recall, precision, label=f"PR-AUC {average_precision_score(target, probability):.3f}"); axes[0, 1].legend(); axes[0, 1].set_title("Precision-recall curve")
    fraction, mean = calibration_curve(target, probability, n_bins=10)
    axes[1, 0].plot(mean, fraction, marker="o"); axes[1, 0].plot([0, 1], [0, 1], "--"); axes[1, 0].set_title("Calibration")
    ranked = pd.DataFrame({"target": target.to_numpy(), "probability": probability}).sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["cumulative_capture"] = ranked["target"].cumsum() / ranked["target"].sum()
    axes[1, 1].plot((ranked.index + 1) / len(ranked), ranked["cumulative_capture"]); axes[1, 1].plot([0, 1], [0, 1], "--"); axes[1, 1].set_title("Cumulative gain"); axes[1, 1].set_xlabel("Share of leads contacted")
    figure.tight_layout()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    figure.savefig(ROOT / "artifacts" / "evaluation_plots.png", dpi=150)
    top_20 = ranked.head(max(1, int(len(ranked) * .20)))
    print(f"Test AUC: {roc_auc_score(target, probability):.3f}")
    print(f"Test PR-AUC: {average_precision_score(target, probability):.3f}")
    print(f"Top 20% capture: {top_20['target'].sum() / ranked['target'].sum():.1%}")


if __name__ == "__main__":
    main()
