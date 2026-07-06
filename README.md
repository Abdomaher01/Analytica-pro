# DEPI Graduation Analytics Project

This project analyzes a social-media crisis dataset with a bronze/silver/gold workflow and an advanced analytics pipeline.

## Project Overview

The workflow is designed to be easy to present, explain, and extend. It includes:

- notebook-based data preparation and exploration
- a production-style training pipeline in advanced_insights.py
- model evaluation and reporting
- clustering, anomaly detection, forecasting, and executive summaries

## File Guide

| File | Purpose |
| --- | --- |
| bronze.ipynb | Loads and prepares the raw dataset. |
| silver.ipynb | Cleans and reshapes the dataset into a reliable analysis table. |
| gold.ipynb | Builds the final analytical outputs. |
| Predictions.ipynb | Runs the full pipeline and reviews reports and metrics. |
| test.ipynb | Scratch notebook for experiments and quick checks. |
| advanced_insights.py | Main training and reporting pipeline. |

## Pipeline Flow

1. Bronze layer: preserve and inspect the starting data.
2. Silver layer: clean data types, handle missing values, and prepare features.
3. Gold layer: create final analytical tables and metrics.
4. Exploratory analysis: understand patterns and relationships in the data.
5. Predictive modeling: train the retained XGBoost models.
6. Advanced insights: generate explanations, clusters, anomalies, forecasts, and summaries.

## Retained Models

The pipeline currently keeps the two strongest supervised models:

- Misinformation Probability — accuracy: 99.9%
- Credibility Score — accuracy: 99.9%

The Engagement Velocity model was removed from the active training workflow because its predictive performance was very low and it did not provide meaningful additional value. It remains available as a feature in the clustering and forecasting workflow, but it is no longer trained as a standalone production model.

## Supporting Analytics

| Component | Algorithm | GPU Support | Purpose |
| --- | --- | --- | --- |
| Segmentation | K-Means and optional HDBSCAN | CPU only | Group behavior patterns |
| Anomaly detection | Isolation Forest | CPU only | Flag unusual records |
| Forecasting | Prophet or RandomForest fallback | CPU only | Project trends over the next 30 days |

Scikit-learn tree ensembles and clustering methods run on CPU because they do not have official GPU training support in this workflow.

## Model Details

- Algorithm: XGBoost Regressor with a reg:squarederror objective
- Preprocessing: numeric features are imputed and scaled; categorical features are imputed and one-hot encoded
- Memory efficiency: numeric columns use float32 and categorical columns use pandas category dtype
- Training approach: train/test split, cross-validation, and randomized hyperparameter search
- Hardware: CUDA is used automatically when available; CPU is used as a fallback
- Explainability: SHAP values and permutation importance highlight the most influential features

## Run the Pipeline

From the project folder:

```bash
python3 advanced_insights.py --data Silver.csv --output-dir .
```

Available options:

- --xgboost-device auto — detect CUDA and fall back to CPU
- --xgboost-device cuda — force GPU training
- --xgboost-device cpu — force CPU training

Training logs are written to outputs/reports/training_environment.txt and outputs/reports/pipeline.log.

## Output Files

The outputs folder is organized into four main areas:

- outputs/reports/: summaries, metrics, and analysis reports
- outputs/plots/: visualizations for presentations and validation
- outputs/datasets/: derived datasets, predictions, clusters, anomalies, and forecasts
- outputs/models/: saved trained models and segmentation/anomaly artifacts
