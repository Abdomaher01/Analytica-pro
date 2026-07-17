# Project Insights Report

## Executive Summary

The workflow retained Misinformation Probability, Credibility Score and used explainability, segmentation, anomaly detection, and forecasting outputs to surface business-relevant insights.

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

- Misinformation Probability is driven most strongly by risk_minus_credibility.
- Credibility Score is driven most strongly by sentiment_score.
- The segmentation workflow identified 3 K-Means clusters with distinct engagement, credibility, toxicity, and misinformation profiles.
- IsolationForest flagged 8781 unusual records that warrant human review because they combine high-risk behavioral signals.
- The forecasted Credibility Score level is 0.7677 over the coming month.
- The forecasted Misinformation Probability level is 0.1796 over the coming month.
- The forecasted Post Volume level is 1268.8909 over the coming month.

## Forecasting Insights

The credibility score forecast is increasing over the next 30 days, with the final projected value near 0.8234.

## Clustering Insights

Cluster profiles should be used to tailor moderation and messaging strategies to different audience segments.

## Anomaly Detection Insights

Anomalies highlight rare combinations of audience size, engagement, toxicity, and misinformation risk that are worth investigating.

## Recommendations

- Prioritize the strongest drivers from the model explanations in monitoring and triage workflows.
- Use the forecast and clustering outputs to prepare response playbooks before risk conditions deteriorate.
- Review anomalies routinely and route high-risk cases to human analysts.

## How to Interpret Results

Use the metrics tables to judge overall fit, the feature importance charts to understand drivers, and the segmentation/anomaly output to guide operational action.

## Visualization Gallery

![Feature Importance](outputs/plots/xgboost/xgboost_feature_importance_credibility_score.png)
![Forecast Credibility](outputs/plots/prophet/forecast_credibility_score.png)
![Forecast Misinformation](outputs/plots/random_forest/forecast_misinformation_probability.png)
