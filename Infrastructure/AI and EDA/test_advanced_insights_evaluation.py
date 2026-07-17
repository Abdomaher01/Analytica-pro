import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path('advanced_insights.py')
spec = importlib.util.spec_from_file_location('advanced_insights', MODULE_PATH)
advanced_insights = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = advanced_insights
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
    """Test that anomaly evaluation returns expected metrics without raising errors."""
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.75, 0.9], dtype=float)
    predictions = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    metrics = advanced_insights.evaluate_anomaly_models(labels, scores, predictions)

    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    assert 'average_precision' in metrics
    assert isinstance(metrics['precision'], float)
    assert isinstance(metrics['recall'], float)



def test_build_markdown_table_formats_dataframes():
    metrics = pd.DataFrame({'metric': ['mae', 'rmse'], 'value': [1.2, 2.5]})

    rendered = advanced_insights.build_markdown_table(metrics, 'Regression Metrics')

    assert 'Regression Metrics' in rendered
    assert '| metric | value |' in rendered
    assert '| mae | 1.2 |' in rendered


def test_load_saved_xgboost_result_ignores_non_numeric_metric_columns(tmp_path):
    paths = advanced_insights.create_output_paths(tmp_path)
    report_dir = paths.reports / 'xgboost'
    dataset_dir = paths.datasets / 'processed'
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {'mae': 0.5, 'rmse': 1.0, 'training_seconds': 2.4, 'device_used': 'cpu'}
    ]).to_csv(report_dir / 'xgboost_metrics_misinformation_probability.csv', index=False)
    pd.DataFrame([{'feature': 'a', 'importance': 0.8}]).to_csv(
        report_dir / 'xgboost_feature_importance_misinformation_probability.csv',
        index=False,
    )
    pd.DataFrame([{'feature': 'a', 'permutation_importance': 0.2, 'directional_effect': 0.1}]).to_csv(
        report_dir / 'xgboost_signed_drivers_misinformation_probability.csv',
        index=False,
    )
    pd.DataFrame({'actual': [0.1, 0.2], 'predicted': [0.15, 0.25]}).to_csv(
        dataset_dir / 'xgboost_predictions_misinformation_probability.csv',
        index=False,
    )
    (paths.models / 'xgboost').mkdir(parents=True, exist_ok=True)
    with open(paths.models / 'xgboost' / 'xgboost_misinformation_probability.pkl', 'wb') as handle:
        pickle.dump({'dummy': 'model'}, handle)

    data = pd.DataFrame({
        'misinformation_probability': [0.1, 0.2, 0.3, 0.4],
        'feature_a': [1.0, 2.0, 3.0, 4.0],
        'feature_b': ['x', 'y', 'x', 'y'],
    })

    result = advanced_insights.load_saved_xgboost_result('misinformation_probability', paths, data, 42)

    assert result is not None
    assert result.metrics['mae'] == 0.5
    assert result.metrics['rmse'] == 1.0
    assert result.device_used == 'cpu'


def test_create_output_paths_uses_model_specific_output_directories(tmp_path):
    paths = advanced_insights.create_output_paths(tmp_path)

    assert (paths.models / 'xgboost').exists()
    assert (paths.plots / 'xgboost').exists()
    assert (paths.reports / 'xgboost').exists()
    assert (paths.datasets / 'forecasting').exists()
    assert (paths.datasets / 'clustering').exists()
    assert (paths.datasets / 'anomaly_detection').exists()
    assert not (tmp_path / 'reports' / 'forecasts').exists()
    assert not (tmp_path / 'assets' / 'images').exists()
