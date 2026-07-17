# DEPI Graduation Analytics Project

## Executive Summary

This project combines explainable predictive modeling, segmentation, anomaly detection, and forecasting for crisis analytics on the Silver dataset.

## Model Performance

### Regression Metrics

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Misinformation Probability | 0.0027 | 0.004 | 0.9996 |
| Credibility Score | 0.0027 | 0.004 | 0.9995 |

### Classification Metrics

| Metric | Value |
| --- | --- |
| Classification metrics | Not available without labeled anomalies |

### Forecasting Metrics

| Metric | Forecast Average | Direction |
| --- | --- | --- |
| Engagement Velocity | 149.1052 | Increase |
| Misinformation Probability | 0.1796 | Increase |
| Post Volume | 1268.8909 | Increase |
| Credibility Score | 0.7677 | Increase |

### Clustering Metrics

| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz | Best |
| --- | --- | --- | --- | --- |
| K-Means | 0.3127 | 1.7403 | 1517.3567 | True |
| Agglomerative Clustering | n/a | n/a | n/a | False |
| DBSCAN | n/a | n/a | n/a | False |

### Anomaly Detection Metrics

| Metric | Value |
| --- | --- |
| anomaly_count | 8781.0 |
| total_records | 50000.0 |
| anomaly_percentage | 17.56 |
| anomaly_ratio | 0.1756 |

## Key Insights & Findings

- The strongest predictive Signals come from the signed driver analysis and SHAP outputs.
- Segmentation reveals recurring audience behavior groups that can be used in targeted moderation.
- Anomaly detection helps identify extreme or suspicious content that deserves analyst attention.

## Forecasting Insights

Credibility-score forecasts and related trend plots are available in the reports and assets folders for stakeholder review.

## Visualization Gallery

![Forecast Credibility](outputs/plots/prophet/forecast_credibility_score.png)
![Feature Importance](outputs/plots/xgboost/xgboost_feature_importance_credibility_score.png)
![Forecast Misinformation](outputs/plots/random_forest/forecast_misinformation_probability.png)

## How to Interpret Results

Use the model metrics for fit, the feature importance charts for driver interpretation, and the forecasting and clustering outputs for business planning.
