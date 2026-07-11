import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path('/home/abdo/Projects/mlearn/DEPI Gradution/Infrastructure/AI and EDA/advanced_insights.py')
spec = importlib.util.spec_from_file_location('advanced_insights', MODULE_PATH)
advanced_insights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advanced_insights)


def test_evaluate_clustering_models_returns_metrics_and_best_model():
    rng = np.random.RandomState(7)
    cluster_a = rng.normal(loc=0.0, scale=0.3, size=(30, 2))
    cluster_b = rng.normal(loc=4.0, scale=0.3, size=(30, 2))
    data = np.vstack([cluster_a, cluster_b])

    results = advanced_insights.evaluate_clustering_models(data, random_state=7)

    assert not results.empty
    expected_columns = {
        'model_name',
        'cluster_count',
        'silhouette_score',
        'davies_bouldin_index',
        'calinski_harabasz_score',
        'inertia',
        'is_best',
    }
    assert expected_columns.issubset(results.columns)
    assert results['is_best'].sum() == 1


def test_evaluate_anomaly_models_with_labels_and_scores():
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.75, 0.9], dtype=float)
    predictions = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    metrics = advanced_insights.evaluate_anomaly_models(labels, scores, predictions)

    assert metrics['precision'] > 0
    assert metrics['recall'] > 0
    assert metrics['f1_score'] > 0
    assert metrics['roc_auc'] >= 0.0
    assert metrics['average_precision'] >= 0.0
