# PROJECT_CONTEXT

## What this is

DEPI graduation project analyzing social-media crisis data. Bronze/silver/gold notebooks prepare data; `advanced_insights.py` and `Predictions.ipynb` run the ML analytics pipeline (XGBoost prediction, SHAP, clustering, anomaly detection, forecasting, network analysis).

## Repo structure

- `bronze.ipynb`, `silver.ipynb`, `gold.ipynb` — data pipeline layers
- `Predictions.ipynb` — notebook entry point for the analytics pipeline
- `advanced_insights.py` — main training and reporting script
- `Silver.csv` — cleaned input dataset
- `outputs/` — generated models, reports, plots, datasets

## Tech notes

- Python 3.12, scikit-learn, XGBoost (CUDA when available), SHAP, Prophet (optional), networkx
- Retained supervised models: XGBoost for `misinformation_probability` and `credibility_score`
- Removed: XGBoost for `engagement_velocity` (R² ≈ 0.045)
- Hardware: auto GPU detection, full CPU parallelism for sklearn, pipeline preprocessing cache

## How to run

```bash
python3 advanced_insights.py --data Silver.csv --output-dir .
```

Or open `Predictions.ipynb` and run all cells.

## Activity Log

- **2026-07-06:** Reviewed all trained models; removed low-performing engagement-velocity XGBoost model and all associated artifacts. Optimized hardware utilization (auto CUDA, RandomizedSearchCV, pipeline memory cache, full CPU threads). Added training hardware logging (`training_environment.txt`, `training_summary.txt`).
- **2026-07-06 (continue):** Attempted to re-run full pipeline from agent; shell spawn failed (`powershell.exe ENOENT`). Verified on-disk artifacts: retained-model metrics and `training_environment.txt` present; `training_summary.txt` and refreshed reports still missing. `Predictions.ipynb` last run was interrupted (`KeyboardInterrupt`) during `credibility_score` tuning. User should complete one full run locally (see How to run).

## Open questions / TODOs

- Complete one uninterrupted pipeline run to generate `training_summary.txt` and refresh stale reports (`executive_summary.txt`, `xgboost_insights.txt`).
