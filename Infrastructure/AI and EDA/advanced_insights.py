"""Advanced analytics pipeline for the social media crisis dataset.
The script reads Silver.csv, creates the outputs folder structure, trains
models, generates reports, and saves publication-ready plots and datasets.
"""
# Importing modules
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    import resource
except ImportError:  # Windows does not provide the resource module
    resource = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from joblib import Memory
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.cluster import HDBSCAN
except Exception:  # pragma: no cover - depends on sklearn version
    HDBSCAN = None

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - environment dependent
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="deep")


TARGETS = [
    "misinformation_probability",
    "credibility_score",
]

# These are the supervised learning targets retained after model evaluation.
# Keeping them in one list lets the same training/evaluation code run for each
# target instead of copying nearly identical modeling blocks.
TARGET_LABELS = {
    "misinformation_probability": "Misinformation Probability",
    "credibility_score": "Credibility Score",
}

# These numeric behavior signals are reused by clustering and anomaly detection.
# Reusing the same feature set makes those two unsupervised analyses easier to
# compare: clusters show common groups, while anomalies show unusual records.
CLUSTER_FEATURES = [
    "follower_count",
    "following_count",
    "account_age_days",
    "likes",
    "shares",
    "comments",
    "engagement_velocity",
    "credibility_score",
    "toxicity_score",
    "misinformation_probability",
]

# Performance tuning: silhouette is O(n^2); Agglomerative is O(n^2) memory/time.
SILHOUETTE_SAMPLE_THRESHOLD = 10_000
SILHOUETTE_SAMPLE_SIZE = 5_000
AGGLOMERATIVE_MAX_SAMPLES = 5_000
CLUSTERING_EVAL_MAX_SAMPLES = 10_000
MAX_SKLEARN_JOBS = 4
MAX_TUNING_JOBS = 4
KMEANS_N_INIT_LARGE = 10
KMEANS_N_INIT_SMALL = 20
CHECKPOINT_FILENAME = "pipeline_checkpoint.json"


@dataclass
class OutputPaths:
    """Centralized output paths so the project is easy to move."""

    root: Path
    models: Path
    plots: Path
    reports: Path
    datasets: Path
    forecast_reports: Path
    assets_images: Path


@dataclass
class ModelResult:
    """Container for trained model artifacts and interpretation tables."""

    target: str
    model: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    signed_effects: pd.DataFrame
    x_test_raw: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray
    training_seconds: float = 0.0
    device_used: str = "cuda"


@dataclass
class ScaledMatrixCache:
    """Reuse imputed/scaled matrices across clustering and anomaly detection."""

    numeric_data: pd.DataFrame
    scaled: np.ndarray


@dataclass
class StageTimer:
    """Track and log elapsed time for every pipeline stage."""

    timings: dict[str, float] = field(default_factory=dict)
    _stack: list[tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        logging.info("Stage started: %s", name)
        self._stack.append((name, started))
        try:
            yield
        finally:
            stage_name, stage_start = self._stack.pop()
            elapsed = time.perf_counter() - stage_start
            self.timings[stage_name] = self.timings.get(stage_name, 0.0) + elapsed
            logging.info("Stage completed: %s in %.2f seconds", stage_name, elapsed)

    def write_report(self, path: Path) -> None:
        lines = ["Pipeline Stage Timings", "=" * 22, ""]
        total = 0.0
        for name, elapsed in self.timings.items():
            lines.append(f"{name}: {elapsed:.2f}s")
            total += elapsed
        lines.extend(["", f"total_logged_seconds: {total:.2f}"])
        save_text(path, "\n".join(lines))


def get_cpu_thread_count() -> int:
    """Return the number of CPU threads available on the host."""

    return max(1, os.cpu_count() or 1)


def get_sklearn_job_count() -> int:
    """Cap sklearn parallelism to avoid nested thread oversubscription."""

    return min(MAX_SKLEARN_JOBS, get_cpu_thread_count())


def get_tuning_job_count(device: str) -> int:
    """Limit hyperparameter search workers; GPU fits must stay serial."""

    if device == "cuda":
        return 1
    return min(MAX_TUNING_JOBS, get_cpu_thread_count())


def get_pipeline_memory() -> Memory:
    """Cache expensive preprocessing steps across hyperparameter search."""

    cache_dir = Path(".cache/sklearn")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Memory(location=str(cache_dir), verbose=0)


def get_training_hardware_profile() -> dict[str, Any]:
    """Collect CPU/GPU information and memory limits for training logs."""

    cpu_count = os.cpu_count() or 1
    cpu_threads = get_cpu_thread_count()
    gpu_info: dict[str, Any] | None = None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, memory_total, memory_used = [
                item.strip() for item in result.stdout.strip().splitlines()[0].split(",", 2)
            ]
            gpu_info = {
                "name": name,
                "memory_total_mb": int(memory_total),
                "memory_used_mb": int(memory_used),
            }
    except Exception:
        gpu_info = None

    peak_rss_mb = 0.0
    if resource is not None:
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss_kb / 1024.0 if os.name != "nt" else peak_rss_kb / (1024.0 * 1024.0)

    return {
        "cpu_count": cpu_count,
        "cpu_threads": cpu_threads,
        "gpu_info": gpu_info,
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def resolve_xgboost_device(requested: str = "cuda") -> str:
    """Prefer CUDA when available and fall back to CPU when needed."""

    if requested == "cpu":
        return "cpu"
    if XGBRegressor is None:
        return "cpu"

    if requested == "cuda":
        candidate = "cuda"
    else:
        candidate = "cuda"

    try:
        probe = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1,
            max_depth=2,
            tree_method="hist",
            device="cuda" if candidate == "cuda" else "cpu",
        )
        probe.fit(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32))
        return candidate
    except Exception as exc:
        if requested == "cuda":
            logging.warning("CUDA requested but unavailable (%s). Falling back to CPU.", exc)
        return "cpu"


def log_training_environment(device: str, paths: OutputPaths) -> None:
    """Log CPU/GPU availability and current memory usage before training."""

    profile = get_training_hardware_profile()
    gpu_name = profile["gpu_info"]["name"] if profile["gpu_info"] else "CPU-only"
    logging.info(
        "Training hardware: device=%s cpu_cores=%s cpu_threads=%s gpu=%s",
        device,
        profile["cpu_count"],
        profile["cpu_threads"],
        gpu_name,
    )
    logging.info("Current memory footprint: %.2f MB", profile["peak_rss_mb"])
    if profile["gpu_info"]:
        logging.info(
            "GPU memory: %s MB used / %s MB total",
            profile["gpu_info"]["memory_used_mb"],
            profile["gpu_info"]["memory_total_mb"],
        )

    save_text(
        paths.reports / "training_environment.txt",
        "\n".join(
            [
                f"device={device}",
                f"cpu_cores={profile['cpu_count']}",
                f"cpu_threads={profile['cpu_threads']}",
                f"gpu={gpu_name}",
                f"peak_rss_mb={profile['peak_rss_mb']:.2f}",
                (
                    f"gpu_memory_used_mb={profile['gpu_info']['memory_used_mb']}"
                    if profile["gpu_info"]
                    else "gpu_memory_used_mb=0"
                ),
                (
                    f"gpu_memory_total_mb={profile['gpu_info']['memory_total_mb']}"
                    if profile["gpu_info"]
                    else "gpu_memory_total_mb=0"
                ),
            ]
        ),
    )


def log_model_training_stats(target: str, device: str, elapsed_seconds: float) -> None:
    """Log per-model hardware usage after training completes."""

    profile = get_training_hardware_profile()
    gpu_name = profile["gpu_info"]["name"] if profile["gpu_info"] else "CPU-only"
    logging.info("Model %s trained in %.2f seconds on %s", target, elapsed_seconds, device)
    logging.info(
        "Model %s resource usage: cpu_threads=%s peak_rss_mb=%.2f gpu=%s",
        target,
        profile["cpu_threads"],
        profile["peak_rss_mb"],
        gpu_name,
    )
    if profile["gpu_info"]:
        logging.info(
            "Model %s GPU memory: %s MB used / %s MB total",
            target,
            profile["gpu_info"]["memory_used_mb"],
            profile["gpu_info"]["memory_total_mb"],
        )


# Status Logging in terminal
def configure_logging(report_dir: Path, append: bool = False) -> None:
    """Log progress to the console and to outputs/reports/pipeline.log."""

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8"),
        ],
        force=True,
    )

# Create output folder structure
def create_output_paths(base_dir: Path) -> OutputPaths:
    """Create the required output folder structure."""

    root = base_dir / "outputs"
    paths = OutputPaths(
        root=root,
        models=root / "models",
        plots=root / "plots",
        reports=root / "reports",
        datasets=root / "datasets",
        forecast_reports=root / "reports" / "forecasting",
        assets_images=root / "plots" / "assets",
    )
    for path in [paths.root, paths.models, paths.plots, paths.reports, paths.datasets]:
        path.mkdir(parents=True, exist_ok=True)

    for model_name in ["xgboost", "random_forest", "prophet", "kmeans", "isolation_forest"]:
        for category_dir in [paths.models, paths.plots, paths.reports]:
            (category_dir / model_name).mkdir(parents=True, exist_ok=True)

    for category in ["processed", "forecasting", "clustering", "anomaly_detection", "feature_engineering"]:
        (paths.datasets / category).mkdir(parents=True, exist_ok=True)

    return paths


def ensure_output_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_output_dir(paths: OutputPaths, model_name: str, artifact_kind: str) -> Path:
    """Return the model-specific output directory for a given artifact kind."""

    if artifact_kind == "models":
        return ensure_output_dir(paths.models / model_name)
    if artifact_kind == "plots":
        return ensure_output_dir(paths.plots / model_name)
    if artifact_kind == "reports":
        return ensure_output_dir(paths.reports / model_name)
    raise ValueError(f"Unsupported artifact kind: {artifact_kind}")


def get_dataset_output_dir(paths: OutputPaths, category: str) -> Path:
    """Return the dataset directory for a given data category."""

    return ensure_output_dir(paths.datasets / category)


# Save reports
def save_text(path: Path, content: str) -> None:
    """Write UTF-8 text reports."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def save_pickle(path: Path, obj: Any) -> None:
    """Persist Python objects without adding non-requested dependencies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def build_markdown_table(dataframe: pd.DataFrame | None, title: str) -> str:
    """Render a simple markdown table from a pandas dataframe."""

    if dataframe is None or dataframe.empty:
        return f"### {title}\n\nNo data available."

    display_frame = dataframe.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].round(4)
        display_frame[column] = display_frame[column].fillna("n/a").astype(str)

    header = "| " + " | ".join(display_frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_frame.columns)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in display_frame.itertuples(index=False, name=None))
    return f"### {title}\n\n{header}\n{separator}\n{body}"


def safe_filename(name: str) -> str:
    """Convert a label into a stable filename token."""

    return name.lower().replace(" ", "_").replace("/", "_")


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Silver dataset and apply defensive type cleanup."""

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find input dataset: {data_path}")

    data = pd.read_csv(data_path)
    logging.info("Loaded %s with shape %s", data_path, data.shape)

    for column in data.columns:
        if column.endswith("_flag") or column in {"is_share"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        data[column] = data[column].astype("float32")

    categorical_columns = [
        column for column in data.columns if column not in numeric_columns and column not in {"post_id", "parent_post_id"}
    ]
    for column in categorical_columns:
        data[column] = data[column].astype("category")

    return data


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create analysis features that help models capture behavior patterns."""

    df = data.copy()
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # Feature setting
    df["total_engagement"] = df[["likes", "shares", "comments"]].sum(axis=1)
    df["share_comment_ratio"] = df["shares"] / (df["comments"] + 1)
    # Reach Indicators
    df["like_rate"] = df["likes"] / (df["impressions_estimate"])
    df["share_rate"] = df["shares"] / (df["impressions_estimate"])
    df["comment_rate"] = df["comments"] / (df["impressions_estimate"])
    df["reach_rate"] = df["reach_estimate"] / (df["impressions_estimate"])
    # Seperate big accounts to small accounts.
    df["audience_size"] = df["follower_count"] + df["following_count"]
    # Detect influential accounts with high follower/following ratio .
    df["follower_following_ratio"] = df["follower_count"] / (df["following_count"] + 1)
    # Make numbers more manageable for models.
    df["log_followers"] = np.log1p(df["follower_count"]) # ln(x + 1)
    df["log_impressions"] = np.log1p(df["impressions_estimate"])
    # Detect posts that are high-risk and the trusted ones.
    df["risk_minus_credibility"] = df["misinformation_probability"] - df["credibility_score"]
    # Detect posts that are toxic and subjective.
    df["toxicity_subjectivity_interaction"] = df["toxicity_score"] * df["subjectivity_score"]

    return df.replace([np.inf, -np.inf], np.nan)


def build_analysis_date(data: pd.DataFrame) -> pd.Series:
    """Create a date column from year, month name, and day name when possible."""

    month_lookup = {
        month: index for index, month in enumerate(pd.date_range("2025-01-01", periods=12, freq="MS").month_name(), start=1)
    }
    month_number = data["month"].map(month_lookup).astype('float64').fillna(1).astype(int)
    year = pd.to_numeric(data["year"], errors="coerce").fillna(pd.Timestamp.today().year).astype(int)

    # The Silver dataset stores weekday names, not day-of-month. Use a stable
    # sequence inside each year-month group to support daily trend analysis.
    sequence_day = data.groupby([year, month_number]).cumcount() + 1
    sequence_day = sequence_day.clip(upper=28)

    return pd.to_datetime(
        {"year": year, "month": month_number, "day": sequence_day},
        errors="coerce",
    )


def get_feature_target_split(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target while avoiding direct leakage."""

    # Leakage happens when the model sees the answer while training. We remove
    # the current target column, plus identifiers that describe rows instead of
    # behavior, before fitting the model.
    excluded_columns = {"post_id", "parent_post_id", target}
    features = data.drop(columns=[column for column in excluded_columns if column in data.columns])
    target_values = pd.to_numeric(data[target], errors="coerce")

    valid_rows = target_values.notna()
    return features.loc[valid_rows].copy(), target_values.loc[valid_rows].copy()


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build automatic numeric and categorical preprocessing."""

    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn versions use the `sparse` argument instead.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Numeric fields are median-imputed and scaled so distance-based methods and
    # model optimization are not dominated by large-count columns.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Categorical fields are filled with the most common value and one-hot
    # encoded so tree models can learn from text/category labels safely.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_xgboost_gpu_params(device: str) -> dict[str, Any]:
    """Return XGBoost parameters for CPU or CUDA training."""

    cpu_threads = get_sklearn_job_count()
    if device == "cpu":
        return {
            "tree_method": "hist",
            "max_bin": 256,
            "max_cat_to_onehot": 4,
            "n_jobs": cpu_threads,
        }

    try:
        import xgboost as xgb
        major_version = int(str(xgb.__version__).split(".")[0])
    except Exception:
        major_version = 2
    if major_version >= 2:
        return {
            "tree_method": "hist",
            "device": "cuda",
            "max_bin": 512,
            "max_cat_to_onehot": 4,
            "predictor": "gpu_predictor",
            "n_jobs": cpu_threads,
        }

    return {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "max_bin": 512,
        "n_jobs": cpu_threads,
    }


def make_xgb_regressor(random_state: int, device: str) -> Any:
    """Create the XGBoost regressor used by the insight engine."""

    if XGBRegressor is None:
        raise ImportError(
            "XGBoost is required for Part 1. Install xgboost and rerun the pipeline."
        ) from XGBOOST_IMPORT_ERROR

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        eval_metric="rmse",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        **get_xgboost_gpu_params(device),
    )


def tune_xgboost_model(
    features: pd.DataFrame,
    target_values: pd.Series,
    random_state: int,
    device: str,
) -> tuple[RandomizedSearchCV, str]:
    """Tune an XGBoost pipeline with cross-validation."""

    active_device = device
    # The Pipeline keeps preprocessing and modeling together. This is important:
    # during cross-validation, preprocessing is learned only from each training
    # fold, which avoids accidentally learning from validation data.
    def build_search(estimator_device: str) -> RandomizedSearchCV:
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(features)),
                ("model", make_xgb_regressor(random_state, estimator_device)),
            ],
            memory=get_pipeline_memory(),
        )

        # RandomizedSearch explores the same space more efficiently than a full
        # grid while keeping model quality high.
        param_distributions = {
            "model__n_estimators": [120, 180, 240],
            "model__max_depth": [4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__subsample": [0.85, 0.9, 1.0],
            "model__colsample_bytree": [0.85, 0.9, 1.0],
            "model__reg_lambda": [0.5, 1.0],
            "model__reg_alpha": [0.0, 0.1],
        }
        folds = KFold(n_splits=2, shuffle=True, random_state=random_state)

        return RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=8,
            scoring="neg_root_mean_squared_error",
            cv=folds,
            n_jobs=get_tuning_job_count(estimator_device),
            random_state=random_state,
            verbose=0,
        )

    search = build_search(active_device)
    started = time.perf_counter()
    try:
        search.fit(features, target_values)
    except Exception as exc:
        if active_device != "cuda":
            raise
        logging.warning("CUDA training failed (%s). Retrying with CPU fallback.", exc)
        active_device = "cpu"
        search = build_search(active_device)
        search.fit(features, target_values)

    elapsed = time.perf_counter() - started
    logging.info("Model tuning completed in %.2f seconds on %s", elapsed, active_device)
    return search, active_device


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate common regression metrics."""

    # MAE is easy to explain, RMSE punishes large mistakes, and R2 shows how
    # much variance the model explains compared with a simple baseline.
    rmse = math.sqrt(mean_squared_error(y_true, predictions))
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": rmse,
        "r2": r2_score(y_true, predictions),
    }


def get_processed_feature_names(model: Pipeline) -> list[str]:
    """Extract feature names after preprocessing."""

    preprocessor = model.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return XGBoost feature importance after preprocessing."""

    feature_names = get_processed_feature_names(model)
    importances = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def estimate_signed_effects(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    device: str = "cpu",
) -> pd.DataFrame:
    """Estimate positive and negative drivers with permutation direction.

    Tree importance is unsigned. To explain directional effects, this function
    measures each raw feature's correlation with predictions and multiplies it
    by permutation importance. Positive scores suggest the feature increases the
    target; negative scores suggest the opposite.
    """

    # Permutation importance answers: "How much worse does the model get if this
    # feature is shuffled?" It works on the full pipeline, so it measures raw
    # input columns instead of only post-encoding feature names.
    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="neg_root_mean_squared_error",
            n_jobs=1 if device == "cuda" else get_sklearn_job_count(),
        )
    except Exception as exc:
        logging.warning("Permutation importance failed (%s). Retrying serially.", exc)
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
    predictions = model.predict(x_test)
    rows = []

    for feature, importance in zip(x_test.columns, result.importances_mean):
        series = x_test[feature]
        if pd.api.types.is_numeric_dtype(series):
            correlation = pd.Series(series).corr(pd.Series(predictions))
        else:
            encoded = series.astype("category").cat.codes.replace(-1, np.nan)
            correlation = encoded.corr(pd.Series(predictions))
        direction = 0 if pd.isna(correlation) else np.sign(correlation)
        rows.append(
            {
                "feature": feature,
                "permutation_importance": importance,
                "directional_effect": importance * direction,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("directional_effect", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance: pd.DataFrame, target: str, paths: OutputPaths) -> None:
    """Save a horizontal feature importance chart."""

    top_features = importance.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature", color="#2f6f9f")
    plt.title(f"Top Feature Importance: {TARGET_LABELS[target]}")
    plt.xlabel("XGBoost importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(get_model_output_dir(paths, "xgboost", "plots") / f"xgboost_feature_importance_{target}.png", dpi=200)
    plt.close()


def plot_prediction_quality(
    y_test: pd.Series,
    predictions: np.ndarray,
    target: str,
    paths: OutputPaths,
) -> None:
    """Save actual-vs-predicted model quality plot."""

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.45, edgecolor=None)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    plt.title(f"Actual vs Predicted: {TARGET_LABELS[target]}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(get_model_output_dir(paths, "xgboost", "plots") / f"xgboost_actual_vs_predicted_{target}.png", dpi=200)
    plt.close()


def load_saved_xgboost_result(
    target: str,
    paths: OutputPaths,
    data: pd.DataFrame,
    random_state: int,
) -> ModelResult | None:
    """Load a previously trained XGBoost model and its evaluation artifacts."""

    model_token = safe_filename(target)
    model_dir = get_model_output_dir(paths, "xgboost", "models")
    report_dir = get_model_output_dir(paths, "xgboost", "reports")
    dataset_dir = get_dataset_output_dir(paths, "processed")
    model_path = model_dir / f"xgboost_{model_token}.pkl"
    metrics_path = report_dir / f"xgboost_metrics_{model_token}.csv"
    importance_path = report_dir / f"xgboost_feature_importance_{model_token}.csv"
    signed_path = report_dir / f"xgboost_signed_drivers_{model_token}.csv"
    predictions_path = dataset_dir / f"xgboost_predictions_{model_token}.csv"

    required_paths = [model_path, metrics_path, importance_path, signed_path, predictions_path]
    if not all(path.exists() for path in required_paths):
        return None

    best_model = load_pickle(model_path)
    metrics_raw = pd.read_csv(metrics_path).iloc[0].to_dict()
    importance = pd.read_csv(importance_path)
    signed_effects = pd.read_csv(signed_path)
    prediction_frame = pd.read_csv(predictions_path)

    features, target_values = get_feature_target_split(data, target)
    _, x_test, _, y_test = train_test_split(
        features,
        target_values,
        test_size=0.2,
        random_state=random_state,
    )
    predictions = prediction_frame["predicted"].to_numpy()

    # Validate and convert metrics with detailed logging
    logging.debug("Loading metrics for target=%s from %s", target, metrics_path)
    logging.debug("Raw metrics dict keys: %s", list(metrics_raw.keys()))
    
    # Extract device_used and training_seconds before processing other metrics
    device_used_raw = metrics_raw.get("device_used", "cuda")
    training_seconds_raw = metrics_raw.get("training_seconds", 0.0)
    
    logging.debug("device_used_raw: value=%r, type=%s", device_used_raw, type(device_used_raw).__name__)
    logging.debug("training_seconds_raw: value=%r, type=%s", training_seconds_raw, type(training_seconds_raw).__name__)
    
    # Validate and convert device_used
    try:
        if pd.isna(device_used_raw) or device_used_raw is None or device_used_raw == "":
            logging.warning("device_used is invalid (NaN/None/empty), using default 'cuda'")
            device_used = "cuda"
        else:
            device_used = str(device_used_raw).strip()
            if not device_used:
                logging.warning("device_used is empty string after strip, using default 'cuda'")
                device_used = "cuda"
    except Exception as exc:
        logging.warning("Failed to convert device_used: %s, using default 'cuda'", exc)
        device_used = "cuda"
    
    logging.debug("device_used validated: value=%r, type=%s", device_used, type(device_used).__name__)
    
    # Validate and convert training_seconds
    try:
        if pd.isna(training_seconds_raw) or training_seconds_raw is None:
            logging.warning("training_seconds is NaN/None, using default 0.0")
            training_seconds = 0.0
        else:
            training_seconds = float(training_seconds_raw)
            if not math.isfinite(training_seconds):
                logging.warning("training_seconds is not finite (inf/nan), using default 0.0")
                training_seconds = 0.0
    except (ValueError, TypeError) as exc:
        logging.warning("Failed to convert training_seconds (value=%r, type=%s): %s, using default 0.0", 
                       training_seconds_raw, type(training_seconds_raw).__name__, exc)
        training_seconds = 0.0
    
    logging.debug("training_seconds validated: value=%r, type=%s", training_seconds, type(training_seconds).__name__)
    
    # Convert numeric metrics only, skipping non-numeric fields
    numeric_metric_keys = {"mae", "rmse", "r2", "best_cv_rmse"}
    metrics_dict = {}
    
    for key, value_raw in metrics_raw.items():
        # Skip device_used and training_seconds - they're handled separately
        if key in ("device_used", "training_seconds"):
            logging.debug("Skipping %s in metrics dict (handled separately)", key)
            continue
        
        # Only convert known numeric metrics
        if key not in numeric_metric_keys:
            logging.debug("Skipping unknown metric key: %s (value=%r)", key, value_raw)
            continue
        
        try:
            if pd.isna(value_raw) or value_raw is None or value_raw == "":
                logging.warning("Metric %s is invalid (NaN/None/empty), using default 0.0", key)
                metrics_dict[key] = 0.0
            else:
                value_float = float(value_raw)
                if not math.isfinite(value_float):
                    logging.warning("Metric %s is not finite (value=%r), using default 0.0", key, value_raw)
                    metrics_dict[key] = 0.0
                else:
                    metrics_dict[key] = value_float
                    logging.debug("Metric %s converted: value=%r, type=%s -> %f", 
                                 key, value_raw, type(value_raw).__name__, value_float)
        except (ValueError, TypeError) as exc:
            logging.warning("Failed to convert metric %s (value=%r, type=%s): %s, using default 0.0", 
                           key, value_raw, type(value_raw).__name__, exc)
            metrics_dict[key] = 0.0
    
    logging.debug("Validated metrics dict: %s", metrics_dict)
    logging.debug("Validated device_used: %s", device_used)
    logging.debug("Validated training_seconds: %f", training_seconds)

    result = ModelResult(
        target=target,
        model=best_model,
        metrics=metrics_dict,
        feature_importance=importance,
        signed_effects=signed_effects,
        x_test_raw=x_test,
        y_test=y_test,
        predictions=predictions,
        training_seconds=training_seconds,
        device_used=device_used,
    )
    
    logging.info("Successfully loaded XGBoost result for %s: metrics_keys=%s, device=%s, training_seconds=%.2f",
                target, list(metrics_dict.keys()), device_used, training_seconds)
    return result


def train_single_xgboost_model(
    target: str,
    data: pd.DataFrame,
    paths: OutputPaths,
    random_state: int,
    device: str,
    data_mtime: float,
    reuse_models: bool,
) -> ModelResult:
    """Train or reuse one XGBoost model with isolated error handling."""

    model_token = safe_filename(target)
    model_path = get_model_output_dir(paths, "xgboost", "models") / f"xgboost_{model_token}.pkl"
    if reuse_models and artifact_is_fresh(model_path, data_mtime):
        cached = load_saved_xgboost_result(target, paths, data, random_state)
        if cached is not None:
            logging.info("Reusing saved XGBoost model for %s", target)
            return cached

    logging.info("Training XGBoost model for %s", target)
    started = time.perf_counter()
    features, target_values = get_feature_target_split(data, target)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target_values,
        test_size=0.2,
        random_state=random_state,
    )

    search, device_used = tune_xgboost_model(x_train, y_train, random_state, device)
    best_model = search.best_estimator_
    predictions = best_model.predict(x_test)
    
    logging.debug("Evaluating regression metrics for target=%s", target)
    metrics = evaluate_regression(y_test, predictions)
    logging.debug("Regression metrics: %s", metrics)
    
    # Validate regression metrics
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            logging.error("Metric %s has invalid type: %s (expected numeric), value=%r", 
                         key, type(value).__name__, value)
        elif pd.isna(value) or not math.isfinite(value):
            logging.error("Metric %s has invalid value: %r (NaN or inf)", key, value)
        else:
            logging.debug("Metric %s validated: %f", key, value)
    
    metrics["best_cv_rmse"] = abs(search.best_score_)
    logging.debug("Added best_cv_rmse: %f", metrics["best_cv_rmse"])

    importance = get_feature_importance(best_model)
    signed_effects = estimate_signed_effects(best_model, x_test, y_test, device_used)

    save_pickle(model_path, best_model)
    
    # Add timing and device info, with validation
    training_time = time.perf_counter() - started
    if not math.isfinite(training_time):
        logging.warning("training_time is not finite, using 0.0")
        training_time = 0.0
    
    metrics["training_seconds"] = training_time
    logging.debug("Added training_seconds: %f", metrics["training_seconds"])
    
    # Validate device_used
    if device_used is None or device_used == "":
        logging.warning("device_used is None or empty, using 'cuda'")
        device_used = "cuda"
    device_used = str(device_used).strip()
    
    metrics["device_used"] = device_used
    logging.debug("Added device_used: %s", metrics["device_used"])
    
    logging.debug("Final metrics dict before saving: %s", {k: (v if k != 'device_used' else v) for k, v in metrics.items()})
    
    report_dir = get_model_output_dir(paths, "xgboost", "reports")
    dataset_dir = get_dataset_output_dir(paths, "processed")
    
    # Save metrics to CSV
    metrics_csv_path = report_dir / f"xgboost_metrics_{model_token}.csv"
    logging.debug("Saving metrics to: %s", metrics_csv_path)
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    logging.debug("Metrics saved successfully")
    
    importance.to_csv(report_dir / f"xgboost_feature_importance_{model_token}.csv", index=False)
    signed_effects.to_csv(report_dir / f"xgboost_signed_drivers_{model_token}.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": predictions}).to_csv(
        dataset_dir / f"xgboost_predictions_{model_token}.csv",
        index=False,
    )

    plot_feature_importance(importance, target, paths)
    plot_prediction_quality(y_test, predictions, target, paths)

    elapsed = time.perf_counter() - started
    result = ModelResult(
        target=target,
        model=best_model,
        metrics=metrics,
        feature_importance=importance,
        signed_effects=signed_effects,
        x_test_raw=x_test,
        y_test=y_test,
        predictions=predictions,
        training_seconds=elapsed,
        device_used=device_used,
    )
    log_model_training_stats(target, device_used, elapsed)
    save_checkpoint(paths, f"xgboost_{target}", {f"xgboost_{target}": str(model_path)})
    return result


def train_xgboost_models(
    data: pd.DataFrame,
    paths: OutputPaths,
    random_state: int,
    device: str,
    checkpoint: dict[str, Any] | None = None,
    data_mtime: float | None = None,
    reuse_models: bool = True,
) -> dict[str, ModelResult]:
    """Train, tune, evaluate, and save one XGBoost model per target."""

    logging.info("Starting XGBoost insight engine")
    logging.info("Parameters: device=%s, random_state=%d, reuse_models=%s, data_mtime=%s",
                device, random_state, reuse_models, data_mtime)
    
    results: dict[str, ModelResult] = {}
    checkpoint = checkpoint or {}
    data_mtime = data_mtime or 0.0

    for target in TARGETS:
        stage_name = f"xgboost_{target}"
        logging.debug("Processing target: %s, stage_name: %s", target, stage_name)
        
        if is_stage_complete(checkpoint, stage_name):
            try:
                logging.info("Stage %s is marked complete in checkpoint, attempting to load cached result", stage_name)
                cached = load_saved_xgboost_result(target, paths, data, random_state)
                if cached is not None:
                    logging.info("Successfully resumed from checkpoint for %s", target)
                    results[target] = cached
                    continue
                else:
                    logging.warning("Stage %s marked complete but could not load saved result, retraining", stage_name)
            except Exception as exc:
                logging.exception("Failed to load cached result for %s: %s", target, exc)
                logging.info("Will retrain model for %s", target)

        try:
            logging.info("Training XGBoost model for target: %s", target)
            results[target] = train_single_xgboost_model(
                target,
                data,
                paths,
                random_state,
                device,
                data_mtime,
                reuse_models=reuse_models,
            )
            logging.info("Successfully trained XGBoost model for %s", target)
        except Exception as exc:
            logging.exception("XGBoost training failed for %s with exception: %s", target, exc)
            import traceback
            full_traceback = traceback.format_exc()
            logging.error("Full traceback:\n%s", full_traceback)
            
            # Attempt to fall back to cached result
            try:
                logging.info("Attempting to fall back to last saved model for %s", target)
                cached = load_saved_xgboost_result(target, paths, data, random_state)
                if cached is not None:
                    logging.warning("Falling back to last saved model for %s", target)
                    results[target] = cached
                else:
                    logging.error("No cached model available for %s, skipping target", target)
            except Exception as fallback_exc:
                logging.exception("Failed to load fallback model for %s: %s", target, fallback_exc)
                logging.error("Skipping target %s due to training failure and no cached model", target)

    if results:
        logging.info("Writing XGBoost reports for %d targets: %s", len(results), list(results.keys()))
        write_xgboost_report(results, paths)
        write_training_summary(results, device, paths)
    else:
        logging.warning("No successful XGBoost training results to report")
    
    return results


def recommendation_for_target(target: str, positive_driver: str, negative_driver: str) -> str:
    """Create a practical recommendation based on model drivers."""

    if target == "misinformation_probability":
        return (
            f"Escalate review queues when {positive_driver} rises, and use patterns like "
            f"{negative_driver} as lower-risk benchmarks."
        )
    return (
        f"Amplify credible content patterns connected to {positive_driver}, and audit "
        f"content conditions where {negative_driver} suppresses credibility."
    )


def write_training_summary(
    results: dict[str, ModelResult],
    device: str,
    paths: OutputPaths,
) -> None:
    """Write per-model training time and hardware usage summary."""

    profile = get_training_hardware_profile()
    gpu_name = profile["gpu_info"]["name"] if profile["gpu_info"] else "CPU-only"
    lines = [
        "Training Summary",
        "================",
        f"resolved_device={device}",
        f"cpu_cores={profile['cpu_count']}",
        f"cpu_threads={profile['cpu_threads']}",
        f"gpu={gpu_name}",
        f"peak_rss_mb={profile['peak_rss_mb']:.2f}",
        "",
    ]
    if profile["gpu_info"]:
        lines.extend(
            [
                f"gpu_memory_used_mb={profile['gpu_info']['memory_used_mb']}",
                f"gpu_memory_total_mb={profile['gpu_info']['memory_total_mb']}",
                "",
            ]
        )

    total_seconds = 0.0
    for target, result in results.items():
        try:
            # Validate result object
            if not isinstance(result, ModelResult):
                logging.error("Result for %s is not a ModelResult: type=%s", target, type(result).__name__)
                continue
            
            # Validate training_seconds
            training_seconds = result.training_seconds
            if not isinstance(training_seconds, (int, float)):
                logging.error("training_seconds for %s has invalid type: %s, using 0.0", 
                             target, type(training_seconds).__name__)
                training_seconds = 0.0
            elif pd.isna(training_seconds) or not math.isfinite(training_seconds):
                logging.error("training_seconds for %s is invalid: %r, using 0.0", target, training_seconds)
                training_seconds = 0.0
            
            # Validate device_used
            device_used = result.device_used
            if not isinstance(device_used, str):
                logging.error("device_used for %s has invalid type: %s, converting to string", 
                             target, type(device_used).__name__)
                device_used = str(device_used)
            
            # Validate metrics
            if not isinstance(result.metrics, dict):
                logging.error("metrics for %s is not a dict: type=%s", target, type(result.metrics).__name__)
                rmse = 0.0
                r2 = 0.0
            else:
                rmse = result.metrics.get('rmse', 0.0)
                r2 = result.metrics.get('r2', 0.0)
                
                # Validate rmse
                if not isinstance(rmse, (int, float)):
                    logging.error("rmse for %s has invalid type: %s, using 0.0", target, type(rmse).__name__)
                    rmse = 0.0
                elif pd.isna(rmse) or not math.isfinite(rmse):
                    logging.error("rmse for %s is invalid: %r, using 0.0", target, rmse)
                    rmse = 0.0
                
                # Validate r2
                if not isinstance(r2, (int, float)):
                    logging.error("r2 for %s has invalid type: %s, using 0.0", target, type(r2).__name__)
                    r2 = 0.0
                elif pd.isna(r2) or not math.isfinite(r2):
                    logging.error("r2 for %s is invalid: %r, using 0.0", target, r2)
                    r2 = 0.0
            
            total_seconds += training_seconds
            lines.extend(
                [
                    TARGET_LABELS[target],
                    f"  device={device_used}",
                    f"  training_seconds={training_seconds:.2f}",
                    f"  rmse={rmse:.6f}",
                    f"  r2={r2:.6f}",
                    "",
                ]
            )
        except Exception as exc:
            logging.exception("Failed to write summary for %s: %s", target, exc)
            lines.extend([
                TARGET_LABELS[target],
                "  ERROR: Failed to generate summary",
                "",
            ])
    
    lines.append(f"total_xgboost_training_seconds={total_seconds:.2f}")
    save_text(get_model_output_dir(paths, "xgboost", "reports") / "training_summary.txt", "\n".join(lines))


def write_xgboost_report(results: dict[str, ModelResult], paths: OutputPaths) -> None:
    """Generate the main XGBoost insight report."""

    sections = ["XGBoost Insight Engine", "=" * 24]
    for target, result in results.items():
        try:
            # Validate result object
            if not isinstance(result, ModelResult):
                logging.error("Result for %s is not a ModelResult: type=%s", target, type(result).__name__)
                sections.extend([
                    "",
                    TARGET_LABELS[target],
                    "ERROR: Invalid result object",
                    "",
                ])
                continue
            
            # Reports convert model artifacts into presentation-ready language:
            # metrics show quality, feature tables show drivers, and recommendations
            # connect those drivers to actions.
            
            # Validate and extract signed_effects
            if not isinstance(result.signed_effects, pd.DataFrame) or result.signed_effects.empty:
                logging.warning("signed_effects for %s is invalid or empty", target)
                positive = pd.DataFrame()
                negative = pd.DataFrame()
                top_positive = "the strongest positive driver"
                top_negative = "the strongest negative driver"
            else:
                positive = result.signed_effects.head(10)
                negative = result.signed_effects.tail(10).sort_values("directional_effect")
                top_positive = positive.iloc[0]["feature"] if not positive.empty else "the strongest positive driver"
                top_negative = negative.iloc[0]["feature"] if not negative.empty else "the strongest negative driver"
            
            # Validate and extract metrics
            if not isinstance(result.metrics, dict):
                logging.error("metrics for %s is not a dict: type=%s", target, type(result.metrics).__name__)
                mae = "N/A"
                rmse = "N/A"
                r2 = "N/A"
                best_cv_rmse = "N/A"
            else:
                mae = result.metrics.get('mae', 0.0)
                rmse = result.metrics.get('rmse', 0.0)
                r2 = result.metrics.get('r2', 0.0)
                best_cv_rmse = result.metrics.get('best_cv_rmse', 0.0)
                
                # Validate each metric
                for metric_name, metric_val in [('mae', mae), ('rmse', rmse), ('r2', r2), ('best_cv_rmse', best_cv_rmse)]:
                    if not isinstance(metric_val, (int, float)):
                        logging.error("%s for %s has invalid type: %s, using 0.0", 
                                     metric_name, target, type(metric_val).__name__)
                        metric_val = 0.0
                    elif pd.isna(metric_val) or not math.isfinite(metric_val):
                        logging.error("%s for %s is invalid: %r, using 0.0", metric_name, target, metric_val)
                        metric_val = 0.0
                    
                    # Update local variables
                    if metric_name == 'mae':
                        mae = metric_val
                    elif metric_name == 'rmse':
                        rmse = metric_val
                    elif metric_name == 'r2':
                        r2 = metric_val
                    elif metric_name == 'best_cv_rmse':
                        best_cv_rmse = metric_val
                
                # Format metrics for display
                mae = f"{mae:.4f}"
                rmse = f"{rmse:.4f}"
                r2 = f"{r2:.4f}"
                best_cv_rmse = f"{best_cv_rmse:.4f}"

            sections.extend(
                [
                    "",
                    TARGET_LABELS[target],
                    "-" * len(TARGET_LABELS[target]),
                    f"MAE: {mae}",
                    f"RMSE: {rmse}",
                    f"R2: {r2}",
                    f"Best cross-validated RMSE: {best_cv_rmse}",
                    "",
                    "Top 20 important features:",
                ]
            )
            
            # Validate and add feature importance
            if isinstance(result.feature_importance, pd.DataFrame) and not result.feature_importance.empty:
                sections.append(result.feature_importance.head(20).to_string(index=False))
            else:
                sections.append("No feature importance data available")
            
            sections.extend([
                "",
                "Strongest positive drivers:",
            ])
            
            if not positive.empty:
                sections.append(positive[["feature", "directional_effect"]].to_string(index=False))
            else:
                sections.append("No positive drivers available")
            
            sections.extend([
                "",
                "Strongest negative drivers:",
            ])
            
            if not negative.empty:
                sections.append(negative[["feature", "directional_effect"]].to_string(index=False))
            else:
                sections.append("No negative drivers available")
            
            sections.extend([
                "",
                "Recommendation:",
                recommendation_for_target(target, str(top_positive), str(top_negative)),
            ])
        except Exception as exc:
            logging.exception("Failed to generate report for %s: %s", target, exc)
            sections.extend([
                "",
                TARGET_LABELS[target],
                "ERROR: Failed to generate report",
                str(exc),
                "",
            ])

    save_text(get_model_output_dir(paths, "xgboost", "reports") / "xgboost_insights.txt", "\n".join(sections))


def run_shap_analysis(results: dict[str, ModelResult], paths: OutputPaths) -> dict[str, pd.DataFrame]:
    """Create SHAP plots and a plain-English explanation report."""

    logging.info("Starting SHAP explainability")
    shap_tables: dict[str, pd.DataFrame] = {}
    sections = ["SHAP Explainability Analysis", "=" * 28]

    if shap is None:
        message = f"SHAP is not available in this environment: {SHAP_IMPORT_ERROR}"
        logging.warning(message)
        save_text(get_model_output_dir(paths, "xgboost", "reports") / "shap_analysis.txt", message)
        return shap_tables

    for target, result in results.items():
        model = result.model
        sample = result.x_test_raw.sample(
            n=min(500, len(result.x_test_raw)),
            random_state=42,
        )
        # SHAP explains the trained XGBoost model after preprocessing. We sample
        # rows to keep the analysis fast while preserving representative cases.
        transformed = model.named_steps["preprocess"].transform(sample)
        feature_names = get_processed_feature_names(model)
        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(transformed_df)

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_table = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_tables[target] = shap_table
        shap_table.to_csv(get_model_output_dir(paths, "xgboost", "reports") / f"shap_importance_{target}.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, transformed_df, show=False, max_display=20)
        plt.title(f"SHAP Summary: {TARGET_LABELS[target]}")
        plt.tight_layout()
        plt.savefig(get_model_output_dir(paths, "xgboost", "plots") / f"shap_summary_{target}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_values, transformed_df, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Bar Importance: {TARGET_LABELS[target]}")
        plt.tight_layout()
        plt.savefig(get_model_output_dir(paths, "xgboost", "plots") / f"shap_bar_{target}.png", dpi=200, bbox_inches="tight")
        plt.close()

        dependence_feature = shap_table.iloc[0]["feature"]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(dependence_feature, shap_values, transformed_df, show=False)
        plt.title(f"SHAP Dependence: {dependence_feature}")
        plt.tight_layout()
        plt.savefig(get_model_output_dir(paths, "xgboost", "plots") / f"shap_dependence_{target}.png", dpi=200, bbox_inches="tight")
        plt.close()

        top_features = ", ".join(shap_table.head(5)["feature"].astype(str))
        sections.extend(
            [
                "",
                TARGET_LABELS[target],
                "-" * len(TARGET_LABELS[target]),
                f"The strongest explanatory signals are: {top_features}.",
                (
                    "These variables explain where the model expects higher or lower "
                    f"{TARGET_LABELS[target].lower()} after accounting for the other fields."
                ),
            ]
        )

    save_text(get_model_output_dir(paths, "xgboost", "reports") / "shap_analysis.txt", "\n".join(sections))
    return shap_tables


def load_checkpoint(paths: OutputPaths) -> dict[str, Any]:
    """Load pipeline checkpoint metadata when resuming interrupted training."""

    checkpoint_path = paths.reports / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return {"completed_stages": [], "artifacts": {}}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(paths: OutputPaths, stage: str, artifacts: dict[str, str] | None = None) -> None:
    """Persist checkpoint after each completed model or pipeline stage."""

    checkpoint_path = paths.reports / CHECKPOINT_FILENAME
    state = load_checkpoint(paths)
    completed = set(state.get("completed_stages", []))
    completed.add(stage)
    state["completed_stages"] = sorted(completed)
    state.setdefault("artifacts", {}).update(artifacts or {})
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    logging.info("Checkpoint saved after stage: %s", stage)


def is_stage_complete(checkpoint: dict[str, Any], stage: str) -> bool:
    """Return True when a pipeline stage was already completed."""

    return stage in checkpoint.get("completed_stages", [])


def artifact_is_fresh(artifact_path: Path, reference_mtime: float) -> bool:
    """Return True when a saved model is newer than the input dataset."""

    return artifact_path.exists() and artifact_path.stat().st_mtime >= reference_mtime


def load_pickle(path: Path) -> Any:
    """Load a pickled artifact."""

    with path.open("rb") as handle:
        return pickle.load(handle)


def compute_silhouette_sampled(
    matrix: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> tuple[float, bool, int]:
    """Compute silhouette on a sample when the dataset exceeds the threshold."""

    n_samples = matrix.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or n_samples < 4:
        return float("nan"), False, n_samples

    if n_samples <= SILHOUETTE_SAMPLE_THRESHOLD:
        return float(silhouette_score(matrix, labels)), False, n_samples

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n_samples)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
    score = float(silhouette_score(matrix[sample_idx], labels[sample_idx]))
    logging.info(
        "Silhouette sampled: n=%s sample=%s (threshold=%s)",
        n_samples,
        sample_size,
        SILHOUETTE_SAMPLE_THRESHOLD,
    )
    return score, True, sample_size


def get_evaluation_subsample(
    matrix: np.ndarray,
    random_state: int,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a representative subsample for expensive clustering algorithms."""

    n_samples = matrix.shape[0]
    if n_samples <= max_samples:
        return matrix, np.arange(n_samples), False

    sample_size = min(max_samples, n_samples)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
    logging.info(
        "Clustering subsample used: n=%s sample=%s max=%s",
        n_samples,
        sample_size,
        max_samples,
    )
    return matrix[sample_idx], sample_idx, True


def get_kmeans_n_init(n_samples: int) -> int:
    """Use fewer K-Means restarts on large datasets."""

    return KMEANS_N_INIT_LARGE if n_samples > SILHOUETTE_SAMPLE_THRESHOLD else KMEANS_N_INIT_SMALL


def prepare_scaled_matrix(data: pd.DataFrame, columns: list[str]) -> ScaledMatrixCache:
    """Impute and scale numeric features for clustering and anomaly detection."""

    available = [column for column in columns if column in data.columns]
    matrix = data[available].apply(pd.to_numeric, errors="coerce")
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(matrix),
        columns=available,
        index=data.index,
    )
    scaled = StandardScaler().fit_transform(imputed)
    return ScaledMatrixCache(numeric_data=imputed, scaled=scaled.astype(np.float32, copy=False))


def _safe_cluster_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    cluster_count: int,
    random_state: int = 42,
) -> dict[str, float | str | int | bool]:
    """Return cluster evaluation metrics when the label structure is valid."""

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(matrix) < 4:
        return {
            "model_name": model_name,
            "cluster_count": int(cluster_count),
            "silhouette_score": np.nan,
            "davies_bouldin_index": np.nan,
            "calinski_harabasz_score": np.nan,
            "inertia": np.nan,
            "silhouette_sampled": False,
            "silhouette_sample_size": 0,
        }

    silhouette, sampled, sample_size = compute_silhouette_sampled(matrix, labels, random_state)
    metrics: dict[str, float | str | int | bool] = {
        "model_name": model_name,
        "cluster_count": int(cluster_count),
        "silhouette_score": silhouette,
        "davies_bouldin_index": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)),
        "silhouette_sampled": sampled,
        "silhouette_sample_size": int(sample_size),
    }
    if model_name == "K-Means":
        metrics["inertia"] = float(np.nan)
    return metrics


def evaluate_clustering_models(
    data: np.ndarray | pd.DataFrame,
    random_state: int = 42,
    paths: OutputPaths | None = None,
    cluster_count: int | None = None,
) -> pd.DataFrame:
    """Evaluate clustering algorithms with sampled metrics on large datasets."""

    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Clustering evaluation expects a 2D feature matrix.")

    if cluster_count is None:
        cluster_count = min(4, max(2, min(matrix.shape[0] - 1, 6)))

    eval_matrix, _, subsampled = get_evaluation_subsample(
        matrix,
        random_state=random_state,
        max_samples=CLUSTERING_EVAL_MAX_SAMPLES,
    )
    n_init = get_kmeans_n_init(matrix.shape[0])
    rows: list[dict[str, float | str | int | bool]] = []

    kmeans_model = KMeans(
        n_clusters=cluster_count,
        n_init=n_init,
        random_state=random_state,
    )
    kmeans_model.fit(eval_matrix)
    kmeans_metrics = _safe_cluster_metrics(
        eval_matrix,
        kmeans_model.labels_,
        "K-Means",
        cluster_count,
        random_state=random_state,
    )
    kmeans_metrics["inertia"] = float(kmeans_model.inertia_)
    rows.append(kmeans_metrics)

    if matrix.shape[0] <= AGGLOMERATIVE_MAX_SAMPLES and not subsampled:
        agglomerative_labels = AgglomerativeClustering(n_clusters=cluster_count).fit_predict(eval_matrix)
        rows.append(
            _safe_cluster_metrics(
                eval_matrix,
                agglomerative_labels,
                "Agglomerative Clustering",
                cluster_count,
                random_state=random_state,
            )
        )
    else:
        logging.info(
            "Skipping Agglomerative Clustering evaluation (n=%s, max=%s).",
            matrix.shape[0],
            AGGLOMERATIVE_MAX_SAMPLES,
        )
        rows.append(
            {
                "model_name": "Agglomerative Clustering",
                "cluster_count": cluster_count,
                "silhouette_score": np.nan,
                "davies_bouldin_index": np.nan,
                "calinski_harabasz_score": np.nan,
                "inertia": np.nan,
                "silhouette_sampled": False,
                "silhouette_sample_size": 0,
                "skipped_reason": "dataset_too_large",
            }
        )

    try:
        dbscan_min_samples = min(max(3, int(eval_matrix.shape[0] * 0.02)), 50)
        dbscan_labels = DBSCAN(eps=0.75, min_samples=dbscan_min_samples).fit_predict(eval_matrix)
        rows.append(
            _safe_cluster_metrics(
                eval_matrix,
                dbscan_labels,
                "DBSCAN",
                len(np.unique(dbscan_labels)),
                random_state=random_state,
            )
        )
    except Exception as exc:
        logging.warning("DBSCAN evaluation failed (%s).", exc)
        rows.append(
            {
                "model_name": "DBSCAN",
                "cluster_count": 0,
                "silhouette_score": np.nan,
                "davies_bouldin_index": np.nan,
                "calinski_harabasz_score": np.nan,
                "inertia": np.nan,
                "silhouette_sampled": False,
                "silhouette_sample_size": 0,
            }
        )

    comparison = pd.DataFrame(rows)
    if "inertia" not in comparison.columns:
        comparison["inertia"] = np.nan
    comparison["is_best"] = False
    ranking = comparison.sort_values(
        ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score"],
        ascending=[False, True, False],
        na_position="last",
    )
    if not ranking.empty and ranking["silhouette_score"].notna().any():
        comparison.loc[ranking.index[0], "is_best"] = True

    if paths is not None:
        report_dir = get_model_output_dir(paths, "kmeans", "reports")
        comparison.to_csv(report_dir / "clustering_model_comparison.csv", index=False)
        with (report_dir / "clustering_model_comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison.to_dict(orient="records"), handle, indent=2)

    return comparison


def choose_kmeans_clusters(scaled: np.ndarray, paths: OutputPaths, random_state: int) -> tuple[int, pd.DataFrame]:
    """Select K using sampled silhouette scores and save elbow diagnostics."""

    max_k = min(10, len(scaled) - 1)
    candidate_ks = list(range(2, max_k + 1))
    selection_matrix, _, _ = get_evaluation_subsample(
        scaled,
        random_state=random_state,
        max_samples=CLUSTERING_EVAL_MAX_SAMPLES,
    )
    n_init = get_kmeans_n_init(scaled.shape[0])
    rows = []

    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = kmeans.fit_predict(selection_matrix)
        silhouette, sampled, sample_size = compute_silhouette_sampled(
            selection_matrix,
            labels,
            random_state=random_state,
        )
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette,
                "silhouette_sampled": sampled,
                "silhouette_sample_size": sample_size,
            }
        )

    scores = pd.DataFrame(rows)
    best_k = int(scores.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    report_dir = get_model_output_dir(paths, "kmeans", "reports")
    plot_dir = get_model_output_dir(paths, "kmeans", "plots")
    scores.to_csv(report_dir / "kmeans_cluster_selection.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=scores, x="k", y="inertia", marker="o")
    plt.title("K-Means Elbow Method")
    plt.tight_layout()
    plt.savefig(plot_dir / "kmeans_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=scores, x="k", y="silhouette_score", marker="o")
    plt.title("K-Means Silhouette Scores")
    plt.tight_layout()
    plt.savefig(plot_dir / "kmeans_silhouette.png", dpi=200)
    plt.close()

    return best_k, scores


def interpret_cluster(row: pd.Series, global_means: pd.Series) -> str:
    """Create a concise behavioral interpretation for one cluster."""

    traits = []
    if row["engagement_velocity"] > global_means["engagement_velocity"]:
        traits.append("high engagement")
    else:
        traits.append("lower engagement")
    if row["misinformation_probability"] > global_means["misinformation_probability"]:
        traits.append("elevated misinformation risk")
    else:
        traits.append("lower misinformation risk")
    if row["credibility_score"] > global_means["credibility_score"]:
        traits.append("above-average credibility")
    else:
        traits.append("weaker credibility")
    if row["toxicity_score"] > global_means["toxicity_score"]:
        traits.append("higher toxicity")

    return "Cluster contains accounts/posts with " + ", ".join(traits) + "."


def run_clustering(
    data: pd.DataFrame,
    paths: OutputPaths,
    random_state: int,
    scaled_cache: ScaledMatrixCache | None = None,
) -> pd.DataFrame:
    """Run HDBSCAN when available and K-Means with automatic K selection."""

    logging.info("Starting user and post segmentation")
    cache = scaled_cache or prepare_scaled_matrix(data, CLUSTER_FEATURES)
    numeric_data = cache.numeric_data
    scaled = cache.scaled
    clustered = data.copy()
    n_init = get_kmeans_n_init(scaled.shape[0])

    best_k, _ = choose_kmeans_clusters(scaled, paths, random_state)
    kmeans = KMeans(n_clusters=best_k, n_init=n_init, random_state=random_state)
    clustered["kmeans_cluster"] = kmeans.fit_predict(scaled)
    save_pickle(get_model_output_dir(paths, "kmeans", "models") / "kmeans_segmentation.pkl", kmeans)

    if HDBSCAN is not None:
        hdbscan_model = HDBSCAN(min_cluster_size=max(10, int(len(data) * 0.02)))
        clustered["hdbscan_cluster"] = hdbscan_model.fit_predict(scaled)
        save_pickle(get_model_output_dir(paths, "kmeans", "models") / "hdbscan_segmentation.pkl", hdbscan_model)
    else:
        clustered["hdbscan_cluster"] = np.nan

    clustered.to_csv(get_dataset_output_dir(paths, "clustering") / "clustered_data.csv", index=False)

    summary = (
        clustered.groupby("kmeans_cluster")
        .agg(
            cluster_size=("post_id", "count"),
            average_engagement=("engagement_velocity", "mean"),
            average_credibility=("credibility_score", "mean"),
            average_toxicity=("toxicity_score", "mean"),
            average_misinformation=("misinformation_probability", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(get_model_output_dir(paths, "kmeans", "reports") / "clustering_summary.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        x=numeric_data["engagement_velocity"],
        y=numeric_data["misinformation_probability"],
        hue=clustered["kmeans_cluster"],
        palette="tab10",
        alpha=0.65,
    )
    plt.title("K-Means Segments: Engagement vs Misinformation")
    plt.tight_layout()
    plt.savefig(get_model_output_dir(paths, "kmeans", "plots") / "clusters_engagement_misinformation.png", dpi=200)
    plt.close()

    global_means = data[CLUSTER_FEATURES].mean(numeric_only=True)
    report_lines = [
        "Clustering Report",
        "=================",
        f"Optimal K-Means clusters selected by silhouette score: {best_k}",
        f"HDBSCAN available: {'yes' if HDBSCAN is not None else 'no'}",
        "",
    ]
    for _, row in summary.iterrows():
        cluster_id = int(row["kmeans_cluster"])
        interpretation_input = pd.Series(
            {
                "engagement_velocity": row["average_engagement"],
                "credibility_score": row["average_credibility"],
                "toxicity_score": row["average_toxicity"],
                "misinformation_probability": row["average_misinformation"],
            }
        )
        report_lines.extend(
            [
                f"Cluster {cluster_id}",
                f"Size: {int(row['cluster_size'])}",
                f"Average engagement: {row['average_engagement']:.4f}",
                f"Average credibility: {row['average_credibility']:.4f}",
                f"Average toxicity: {row['average_toxicity']:.4f}",
                f"Average misinformation: {row['average_misinformation']:.4f}",
                interpret_cluster(interpretation_input, global_means),
                "",
            ]
        )

    clustering_comparison = evaluate_clustering_models(scaled, random_state=random_state, paths=paths, cluster_count=best_k)
    comparison_lines = [
        "",
        "Clustering Performance Evaluation",
        "================================",
        "This section compares clustering algorithms using silhouette, Davies-Bouldin, Calinski-Harabasz, and inertia metrics.",
        "Higher silhouette and Calinski-Harabasz scores are better; lower Davies-Bouldin and inertia are better.",
        clustering_comparison.to_string(index=False),
        "",
        f"Best-performing clustering model: {clustering_comparison.loc[clustering_comparison['is_best'], 'model_name'].iloc[0]}",
    ]
    report_lines.extend(comparison_lines)
    save_text(get_model_output_dir(paths, "kmeans", "reports") / "clustering_report.txt", "\n".join(report_lines))
    return clustered


def explain_anomaly_reasons(row: pd.Series, thresholds: pd.Series) -> str:
    """Explain why IsolationForest may have flagged a record."""

    reasons = []
    if row["engagement_velocity"] > thresholds["engagement_velocity"]:
        reasons.append("abnormal engagement spike")
    if row["misinformation_probability"] > thresholds["misinformation_probability"]:
        reasons.append("high misinformation probability")
    if row["toxicity_score"] > thresholds["toxicity_score"]:
        reasons.append("high toxicity")
    if row["follower_count"] > thresholds["follower_count"]:
        reasons.append("unusually large audience")
    if not reasons:
        reasons.append("unusual multivariate behavior across account and post metrics")
    return "; ".join(reasons)


def evaluate_anomaly_models(
    true_labels: np.ndarray | pd.Series | None,
    anomaly_scores: np.ndarray | pd.Series,
    predicted_labels: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate anomaly detection performance when ground-truth labels are available."""

    if true_labels is None:
        raise ValueError("Ground-truth labels are required for supervised anomaly evaluation.")

    y_true = np.asarray(true_labels).ravel()
    scores = np.asarray(anomaly_scores).ravel()
    if predicted_labels is None:
        predicted_labels = np.where(scores >= np.median(scores), 1, 0)
    y_pred = np.asarray(predicted_labels).ravel()

    if y_true.dtype.kind in {"i", "u", "f"}:
        if -1 in y_true or -1 in y_pred:
            y_true_binary = np.where(y_true == -1, 1, 0)
            y_pred_binary = np.where(y_pred == -1, 1, 0)
        else:
            y_true_binary = y_true
            y_pred_binary = y_pred
    else:
        y_true_binary = np.where(np.asarray(y_true) == "anomaly", 1, 0)
        y_pred_binary = np.where(np.asarray(y_pred) == "anomaly", 1, 0)

    if y_pred_binary.ndim != 1:
        y_pred_binary = y_pred_binary.ravel()

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    confusion = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    roc_auc = float("nan")
    average_precision = float("nan")

    if len(np.unique(y_true_binary)) == 2 and len(np.unique(scores)) > 1:
        roc_auc = roc_auc_score(y_true_binary, scores)
        average_precision = average_precision_score(y_true_binary, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": confusion,
    }


def vectorized_anomaly_reasons(
    numeric_data: pd.DataFrame,
    labels: np.ndarray,
    thresholds: pd.Series,
) -> np.ndarray:
    """Explain anomaly flags with vectorized threshold checks."""

    reasons = np.full(len(labels), "normal behavior", dtype=object)
    anomaly_mask = labels == -1
    if not anomaly_mask.any():
        return reasons

    checks = {
        "engagement_velocity": "abnormal engagement spike",
        "misinformation_probability": "high misinformation probability",
        "toxicity_score": "high toxicity",
        "follower_count": "unusually large audience",
    }
    reason_columns: list[pd.Series] = []
    for column, phrase in checks.items():
        if column in numeric_data.columns and column in thresholds.index:
            reason_columns.append(
                pd.Series(
                    np.where(numeric_data[column].to_numpy() > thresholds[column], phrase, ""),
                    index=numeric_data.index,
                )
            )

    if reason_columns:
        combined = reason_columns[0]
        for column in reason_columns[1:]:
            both = (combined != "") & (column != "")
            combined = combined.where(~both, combined + "; " + column)
            combined = combined.where(combined != "", column)
        fallback = "unusual multivariate behavior across account and post metrics"
        combined = combined.replace("", fallback)
        reasons[anomaly_mask] = combined.to_numpy()[anomaly_mask]
    else:
        reasons[anomaly_mask] = "unusual multivariate behavior across account and post metrics"

    return reasons


def evaluate_unsupervised_anomaly_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    contamination: str | float = "auto",
) -> dict[str, float]:
    """Summarize anomaly detection when ground-truth labels are unavailable."""

    anomaly_mask = labels == -1
    anomaly_count = int(anomaly_mask.sum())
    total_records = int(len(labels))
    return {
        "anomaly_count": anomaly_count,
        "total_records": total_records,
        "anomaly_percentage": round((anomaly_count / total_records) * 100 if total_records else 0.0, 2),
        "contamination_setting": float(contamination) if isinstance(contamination, (int, float)) else float("nan"),
        "anomaly_ratio": round(anomaly_count / total_records, 4) if total_records else 0.0,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std": float(scores.std()),
        "score_p25": float(np.percentile(scores, 25)),
        "score_p75": float(np.percentile(scores, 75)),
    }


def run_anomaly_detection(
    data: pd.DataFrame,
    paths: OutputPaths,
    random_state: int,
    scaled_cache: ScaledMatrixCache | None = None,
) -> pd.DataFrame:
    """Detect suspicious accounts, unusual posts, and engagement spikes."""

    logging.info("Starting anomaly detection")
    cache = scaled_cache or prepare_scaled_matrix(data, CLUSTER_FEATURES)
    numeric_data = cache.numeric_data
    scaled = cache.scaled
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=get_sklearn_job_count(),
    )
    labels = model.fit_predict(scaled)
    scores = model.decision_function(scaled)
    save_pickle(get_model_output_dir(paths, "isolation_forest", "models") / "isolation_forest_anomaly_detector.pkl", model)

    scored = data.copy()
    scored["anomaly_label"] = labels
    scored["anomaly_score"] = scores
    thresholds = numeric_data.quantile(0.95)
    scored["anomaly_reason"] = vectorized_anomaly_reasons(numeric_data, labels, thresholds)

    anomalies = scored[scored["anomaly_label"] == -1].sort_values("anomaly_score")
    anomalies.to_csv(get_dataset_output_dir(paths, "anomaly_detection") / "anomalies.csv", index=False)

    stats = evaluate_unsupervised_anomaly_metrics(labels, scores, contamination="auto")
    report_dir = get_model_output_dir(paths, "isolation_forest", "reports")
    plot_dir = get_model_output_dir(paths, "isolation_forest", "plots")
    pd.DataFrame([stats]).to_csv(report_dir / "anomaly_metrics.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=scored,
        x="engagement_velocity",
        y="misinformation_probability",
        hue=scored["anomaly_label"].map({1: "normal", -1: "anomaly"}),
        alpha=0.65,
    )
    plt.title("Anomaly Detection: Engagement vs Misinformation")
    plt.tight_layout()
    plt.savefig(plot_dir / "anomalies_engagement_misinformation.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(scored["anomaly_score"], kde=True, bins=20)
    plt.title("IsolationForest Anomaly Score Distribution")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / "anomaly_score_distribution.png", dpi=200)
    plt.close()

    top_reasons = anomalies["anomaly_reason"].value_counts().head(10)

    report_lines = [
        "Anomaly Detection Report",
        "========================",
        f"Flagged records: {len(anomalies):,} of {len(scored):,}",
        f"Detected anomaly percentage: {stats['anomaly_percentage']:.2f}%",
        f"Anomaly score statistics: min={stats['score_min']:.4f}, max={stats['score_max']:.4f}, mean={stats['score_mean']:.4f}, median={stats['score_median']:.4f}, std={stats['score_std']:.4f}",
        "",
        "Most common anomaly reasons:",
        top_reasons.to_string(),
        "",
        "Records are flagged by IsolationForest when their combined account, engagement, credibility, toxicity, and misinformation profile is rare compared with the rest of the dataset.",
        "",
        "Anomaly Detection Performance Evaluation",
        "======================================",
        "Supervised anomaly metrics are skipped unless labeled anomalies are provided in the dataset.",
    ]

    ground_truth_column = None
    for candidate in ["is_anomaly", "anomaly_ground_truth", "ground_truth_anomaly", "anomaly_true", "ground_truth_label"]:
        if candidate in data.columns:
            ground_truth_column = candidate
            break

    if ground_truth_column is not None:
        ground_truth = pd.to_numeric(data[ground_truth_column], errors="coerce").fillna(0)
        metrics = evaluate_anomaly_models(ground_truth, scored["anomaly_score"], scored["anomaly_label"])
        confusion = pd.DataFrame(metrics["confusion_matrix"], index=["normal", "anomaly"], columns=["predicted_normal", "predicted_anomaly"])
        confusion.to_csv(report_dir / "anomaly_confusion_matrix.csv")
        pd.DataFrame([metrics]).to_csv(report_dir / "anomaly_supervised_metrics.csv", index=False)
        report_lines.extend(
            [
                "Supervised evaluation metrics:",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1-score: {metrics['f1_score']:.4f}",
                f"ROC-AUC: {metrics['roc_auc']:.4f}",
                f"Average Precision: {metrics['average_precision']:.4f}",
                "",
                "Confusion matrix:",
                confusion.to_string(),
            ]
        )
        if len(np.unique(ground_truth)) > 1 and len(np.unique(scored["anomaly_score"])) > 1:
            fpr, tpr, _ = roc_curve(np.where(ground_truth == -1, 1, 0), scored["anomaly_score"])
            precision_values, recall_values, _ = precision_recall_curve(np.where(ground_truth == -1, 1, 0), scored["anomaly_score"])
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label="ROC")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig(plot_dir / "anomaly_roc_curve.png", dpi=200)
            plt.close()

            plt.figure(figsize=(6, 5))
            plt.plot(recall_values, precision_values, label="Precision-Recall")
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(plot_dir / "anomaly_precision_recall_curve.png", dpi=200)
            plt.close()

            report_lines.extend(["ROC and precision-recall curves were saved to the plots directory."])
    else:
        report_lines.extend([
            "No labeled anomaly column was found, so supervised metrics could not be computed.",
            "The unsupervised summary above captures the anomaly count, scores, and reasons instead.",
        ])

    save_text(report_dir / "anomaly_report.txt", "\n".join(report_lines))
    return scored


def aggregate_daily_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics into a daily time-series frame."""

    df = data.copy()
    df["analysis_date"] = build_analysis_date(df)
    # Forecasting needs one row per time period. Here we compress many posts into
    # daily averages plus daily post volume.
    return (
        df.groupby("analysis_date")
        .agg(
            engagement_velocity=("engagement_velocity", "mean"),
            misinformation_probability=("misinformation_probability", "mean"),
            credibility_score=("credibility_score", "mean"),
            post_volume=("post_id", "count"),
        )
        .reset_index()
        .sort_values("analysis_date")
    )


def forecast_with_prophet(series: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Forecast a single metric with Prophet."""

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(series.rename(columns={"analysis_date": "ds", "value": "y"}))
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "analysis_date"})


def forecast_with_sklearn(series: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Forecast with a lightweight trend model when Prophet is unavailable."""

    series = series.sort_values("analysis_date").copy()
    series["time_index"] = np.arange(len(series), dtype=np.float32)
    model = RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=get_sklearn_job_count())
    model.fit(series[["time_index"]], series["value"])

    future_index = np.arange(len(series) + periods, dtype=np.float32)
    future_dates = pd.date_range(series["analysis_date"].min(), periods=len(series) + periods, freq="D")
    predictions = model.predict(pd.DataFrame({"time_index": future_index}))
    residual = series["value"] - model.predict(series[["time_index"]])
    interval = 1.96 * residual.std(ddof=0)

    return pd.DataFrame(
        {
            "analysis_date": future_dates,
            "yhat": predictions.astype(float),
            "yhat_lower": (predictions - interval).astype(float),
            "yhat_upper": (predictions + interval).astype(float),
        }
    )


def save_forecast_plot(series: pd.DataFrame, forecast: pd.DataFrame, metric: str, paths: OutputPaths) -> None:
    """Persist forecast plots to the standardized plots directory for the selected forecasting model."""

    model_name = "prophet" if Prophet is not None else "random_forest"
    plot_dir = get_model_output_dir(paths, model_name, "plots")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=series, x="analysis_date", y="value", label="actual", ax=ax)
    sns.lineplot(data=forecast, x="analysis_date", y="yhat", label="forecast", ax=ax)
    ax.fill_between(
        forecast["analysis_date"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.18,
        color="C1",
    )
    ax.set_title(f"30-Day Forecast: {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(plot_dir / f"forecast_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_forecasting(data: pd.DataFrame, paths: OutputPaths, periods: int = 30) -> dict[str, pd.DataFrame]:
    """Forecast engagement, misinformation, post volume, and credibility for 30 days."""

    logging.info("Starting time-series forecasting")
    daily = aggregate_daily_metrics(data)
    daily.to_csv(get_dataset_output_dir(paths, "forecasting") / "daily_metrics.csv", index=False)

    forecasts = {}
    report_lines = [
        "Forecast Report",
        "===============",
        f"Forecast horizon: {periods} days",
        f"Method: {'Prophet' if Prophet is not None else 'RandomForest trend fallback'}",
        "",
    ]

    for metric in ["engagement_velocity", "misinformation_probability", "credibility_score", "post_volume"]:
        series = daily[["analysis_date", metric]].rename(columns={metric: "value"}).dropna()
        # Prophet is preferred when installed because it is designed for dated
        # observations; otherwise the RandomForest fallback keeps the workflow
        # usable in simpler environments.
        if Prophet is not None and len(series) >= 10:
            forecast = forecast_with_prophet(series, periods)
        else:
            forecast = forecast_with_sklearn(series, periods)

        forecast["metric"] = metric
        forecasts[metric] = forecast
        forecast.to_csv(get_dataset_output_dir(paths, "forecasting") / f"forecast_{metric}.csv", index=False)

        save_forecast_plot(series, forecast, metric, paths)

        recent_mean = series.tail(min(7, len(series)))["value"].mean()
        future_mean = forecast.tail(periods)["yhat"].mean()
        direction = "increase" if future_mean > recent_mean else "decline"
        report_lines.extend(
            [
                metric.replace("_", " ").title(),
                f"Recent average: {recent_mean:.4f}",
                f"Forecast average next {periods} days: {future_mean:.4f}",
                f"Interpretation: expected to {direction} versus the recent baseline.",
                "",
            ]
        )

    forecast_report_dir = get_model_output_dir(paths, "prophet" if Prophet is not None else "random_forest", "reports")
    save_text(forecast_report_dir / "forecast_report.txt", "\n".join(report_lines))
    return forecasts


def summarize_top_driver(results: dict[str, ModelResult], target: str) -> str:
    """Return the strongest positive driver for executive reporting."""

    if target not in results or results[target].signed_effects.empty:
        return "the strongest modeled driver"
    return str(results[target].signed_effects.iloc[0]["feature"])


def write_executive_summary(
    model_results: dict[str, ModelResult],
    clustered: pd.DataFrame,
    anomalies: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    paths: OutputPaths,
) -> None:
    """Generate a stakeholder-ready business summary."""

    # This final report translates technical outputs into decision questions.
    # It is meant for slides or stakeholder discussion, not model debugging.
    cluster_count = clustered["kmeans_cluster"].nunique()
    anomaly_count = len(anomalies)
    misinformation_driver = summarize_top_driver(model_results, "misinformation_probability")
    credibility_driver = summarize_top_driver(model_results, "credibility_score")

    def forecast_direction(metric: str) -> str:
        forecast = forecasts.get(metric)
        if forecast is None or forecast.empty:
            return "stable"
        first = forecast.tail(30)["yhat"].iloc[0]
        last = forecast.tail(30)["yhat"].iloc[-1]
        return "upward" if last > first else "downward"

    retained_targets = ", ".join(TARGET_LABELS[target] for target in model_results)
    summary = f"""
Executive Summary
=================

1. Which predictive models were retained?
The retained workflow focuses on the strongest targets: {retained_targets}. The engagement-velocity target was removed from the production pipeline because it did not achieve a stable predictive fit and offered limited additional value over the retained models.

2. What drives misinformation?
Misinformation risk is most strongly associated with {misinformation_driver}. This signal should be prioritized in moderation queues and early-warning dashboards.

3. What characterizes credible content?
Credibility is most strongly associated with {credibility_driver}. Stakeholders should amplify content patterns that match this profile and audit posts that move in the opposite direction.

4. Which user groups exist?
The segmentation model identified {cluster_count} K-Means user/post groups. These groups separate audiences by engagement, credibility, toxicity, and misinformation risk, enabling differentiated response strategies.

5. Which anomalies were discovered?
IsolationForest flagged {anomaly_count:,} unusual records. These records show rare combinations of audience size, engagement, toxicity, credibility, and misinformation risk, making them strong candidates for human review.

6. What trends are expected next month?
The forecast suggests an {forecast_direction('misinformation_probability')} misinformation trend, an {forecast_direction('credibility_score')} credibility trend, and an {forecast_direction('post_volume')} post-volume trend over the next 30 days.

7. What are the most important strategic recommendations?
- Build monitoring dashboards around the strongest misinformation and credibility drivers.
- Prioritize anomalous high-risk records for analyst review.
- Use cluster profiles to tailor crisis response messages by audience behavior.
- Monitor engagement and credibility trends to detect emerging crises.
- Re-run this pipeline after each major crisis phase to keep forecasts and model explanations current.
"""
    save_text(paths.reports / "executive_summary.txt", summary)


def write_project_insights_report(
    output_dir: Path,
    paths: OutputPaths,
    model_results: dict[str, ModelResult],
    clustered: pd.DataFrame | None,
    anomalies: pd.DataFrame | None,
    forecasts: dict[str, pd.DataFrame],
    clustering_comparison: pd.DataFrame | None = None,
    anomaly_metrics: pd.DataFrame | None = None,
    supervised_anomaly_metrics: pd.DataFrame | None = None,
) -> None:
    """Generate stakeholder-ready markdown summaries and save them to the reports and README."""

    report_lines = ["# Project Insights Report", "", "## Executive Summary", ""]
    if model_results:
        retained_targets = ", ".join(TARGET_LABELS[target] for target in model_results)
        report_lines.append(
            f"The workflow retained {retained_targets} and used explainability, segmentation, anomaly detection, and forecasting outputs to surface business-relevant insights."
        )
    else:
        report_lines.append("The workflow completed without an active model set, so no supervised model summary is available yet.")

    report_lines.extend(["", "## Model Performance", ""])
    regression_rows = []
    for target, result in model_results.items():
        try:
            # Validate result object and metrics
            if not isinstance(result, ModelResult):
                logging.error("Result for %s is not a ModelResult: type=%s", target, type(result).__name__)
                mae = float("nan")
                rmse = float("nan")
                r2 = float("nan")
            elif not isinstance(result.metrics, dict):
                logging.error("metrics for %s is not a dict: type=%s", target, type(result.metrics).__name__)
                mae = float("nan")
                rmse = float("nan")
                r2 = float("nan")
            else:
                mae = result.metrics.get("mae", float("nan"))
                rmse = result.metrics.get("rmse", float("nan"))
                r2 = result.metrics.get("r2", float("nan"))
                
                # Validate each metric value
                for metric_name, metric_val in [('mae', mae), ('rmse', rmse), ('r2', r2)]:
                    if isinstance(metric_val, str):
                        logging.error("%s for %s is a string: %r, using nan", metric_name, target, metric_val)
                        metric_val = float("nan")
                    elif not isinstance(metric_val, (int, float)):
                        logging.error("%s for %s has invalid type: %s, using nan", 
                                     metric_name, target, type(metric_val).__name__)
                        metric_val = float("nan")
                    
                    # Update local variables
                    if metric_name == 'mae':
                        mae = metric_val
                    elif metric_name == 'rmse':
                        rmse = metric_val
                    elif metric_name == 'r2':
                        r2 = metric_val
            
            regression_rows.append(
                {
                    "Model": TARGET_LABELS[target],
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )
        except Exception as exc:
            logging.exception("Failed to extract metrics for %s: %s", target, exc)
            regression_rows.append(
                {
                    "Model": TARGET_LABELS[target],
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "R2": float("nan"),
                }
            )
    report_lines.append(build_markdown_table(pd.DataFrame(regression_rows), "Regression Metrics"))

    classification_rows = []
    if supervised_anomaly_metrics is not None and not supervised_anomaly_metrics.empty:
        classification_rows.append(
            {
                "Metric": "Precision",
                "Value": supervised_anomaly_metrics.iloc[0].get("precision", float("nan")),
            }
        )
        classification_rows.append(
            {
                "Metric": "Recall",
                "Value": supervised_anomaly_metrics.iloc[0].get("recall", float("nan")),
            }
        )
        classification_rows.append(
            {
                "Metric": "F1 Score",
                "Value": supervised_anomaly_metrics.iloc[0].get("f1_score", float("nan")),
            }
        )
        classification_rows.append(
            {
                "Metric": "ROC-AUC",
                "Value": supervised_anomaly_metrics.iloc[0].get("roc_auc", float("nan")),
            }
        )
    else:
        classification_rows.append({"Metric": "Classification metrics", "Value": "Not available without labeled anomalies"})
    report_lines.append("")
    report_lines.append(build_markdown_table(pd.DataFrame(classification_rows), "Classification Metrics"))

    forecast_rows = []
    for metric in ["engagement_velocity", "misinformation_probability", "post_volume", "credibility_score"]:
        if metric in forecasts and not forecasts[metric].empty:
            future_mean = forecasts[metric].tail(30)["yhat"].mean()
            recent_mean = 0.0
            if metric in {"engagement_velocity", "misinformation_probability", "credibility_score"} and (paths.datasets / "daily_metrics.csv").exists():
                daily_metrics = pd.read_csv(paths.datasets / "daily_metrics.csv")
                if metric in daily_metrics.columns:
                    recent_mean = float(daily_metrics[metric].tail(7).mean())
            forecast_rows.append({
                "Metric": metric.replace("_", " ").title(),
                "Forecast Average": future_mean,
                "Direction": "Increase" if future_mean > recent_mean else "Decline",
            })
    report_lines.append("")
    report_lines.append(build_markdown_table(pd.DataFrame(forecast_rows), "Forecasting Metrics"))

    clustering_rows = []
    if clustering_comparison is not None and not clustering_comparison.empty:
        for _, row in clustering_comparison.iterrows():
            clustering_rows.append(
                {
                    "Model": row.get("model_name", "n/a"),
                    "Silhouette": row.get("silhouette_score", float("nan")),
                    "Davies-Bouldin": row.get("davies_bouldin_index", float("nan")),
                    "Calinski-Harabasz": row.get("calinski_harabasz_score", float("nan")),
                    "Best": row.get("is_best", False),
                }
            )
    report_lines.append("")
    report_lines.append(build_markdown_table(pd.DataFrame(clustering_rows), "Clustering Metrics"))

    anomaly_rows = []
    if anomaly_metrics is not None and not anomaly_metrics.empty:
        for column in ["anomaly_count", "total_records", "anomaly_percentage", "anomaly_ratio"]:
            if column in anomaly_metrics.columns:
                anomaly_rows.append({"Metric": column, "Value": anomaly_metrics.iloc[0][column]})
    else:
        anomaly_rows.append({"Metric": "anomaly_count", "Value": len(anomalies) if anomalies is not None else 0})
    report_lines.append("")
    report_lines.append(build_markdown_table(pd.DataFrame(anomaly_rows), "Anomaly Detection Metrics"))

    report_lines.extend(["", "## Key Insights & Findings", ""])
    if model_results:
        feature_lines = []
        for target, result in model_results.items():
            feature_lines.append(f"- {TARGET_LABELS[target]} is driven most strongly by {result.signed_effects.iloc[0]['feature'] if not result.signed_effects.empty else 'the strongest available signal'}.")
        report_lines.extend(feature_lines)

    if clustered is not None and not clustered.empty:
        cluster_count = int(clustered["kmeans_cluster"].nunique())
        report_lines.append(f"- The segmentation workflow identified {cluster_count} K-Means clusters with distinct engagement, credibility, toxicity, and misinformation profiles.")

    if anomalies is not None and not anomalies.empty:
        report_lines.append(f"- IsolationForest flagged {len(anomalies)} unusual records that warrant human review because they combine high-risk behavioral signals.")

    if forecasts:
        for metric in ["credibility_score", "misinformation_probability", "post_volume"]:
            if metric in forecasts and not forecasts[metric].empty:
                trend = forecasts[metric].tail(30)["yhat"].mean()
                report_lines.append(f"- The forecasted {metric.replace('_', ' ').title()} level is {trend:.4f} over the coming month.")

    report_lines.extend(["", "## Forecasting Insights", ""])
    credibility_forecast = forecasts.get("credibility_score")
    if credibility_forecast is not None and not credibility_forecast.empty:
        latest = credibility_forecast["yhat"].iloc[-1]
        first = credibility_forecast["yhat"].iloc[0]
        direction = "increasing" if latest > first else "decreasing"
        report_lines.append(f"The credibility score forecast is {direction} over the next 30 days, with the final projected value near {latest:.4f}.")
    report_lines.extend(["", "## Clustering Insights", ""])
    report_lines.append("Cluster profiles should be used to tailor moderation and messaging strategies to different audience segments.")
    report_lines.extend(["", "## Anomaly Detection Insights", ""])
    report_lines.append("Anomalies highlight rare combinations of audience size, engagement, toxicity, and misinformation risk that are worth investigating.")
    report_lines.extend(["", "## Recommendations", ""])
    report_lines.append("- Prioritize the strongest drivers from the model explanations in monitoring and triage workflows.")
    report_lines.append("- Use the forecast and clustering outputs to prepare response playbooks before risk conditions deteriorate.")
    report_lines.append("- Review anomalies routinely and route high-risk cases to human analysts.")
    report_lines.extend(["", "## How to Interpret Results", ""])
    report_lines.append("Use the metrics tables to judge overall fit, the feature importance charts to understand drivers, and the segmentation/anomaly output to guide operational action.")
    report_lines.extend(["", "## Visualization Gallery", ""])
    report_lines.append("![Feature Importance](outputs/plots/xgboost/xgboost_feature_importance_credibility_score.png)")
    report_lines.append("![Forecast Credibility](outputs/plots/prophet/forecast_credibility_score.png)")
    report_lines.append("![Forecast Misinformation](outputs/plots/random_forest/forecast_misinformation_probability.png)")

    report_text = "\n".join(report_lines) + "\n"
    save_text(paths.reports / "insights_report.md", report_text)

    readme_lines = [
        "# DEPI Graduation Analytics Project",
        "",
        "## Executive Summary",
        "",
        "This project combines explainable predictive modeling, segmentation, anomaly detection, and forecasting for crisis analytics on the Silver dataset.",
        "",
        "## Model Performance",
        "",
        build_markdown_table(pd.DataFrame(regression_rows), "Regression Metrics"),
        "",
        build_markdown_table(pd.DataFrame(classification_rows), "Classification Metrics"),
        "",
        build_markdown_table(pd.DataFrame(forecast_rows), "Forecasting Metrics"),
        "",
        build_markdown_table(pd.DataFrame(clustering_rows), "Clustering Metrics"),
        "",
        build_markdown_table(pd.DataFrame(anomaly_rows), "Anomaly Detection Metrics"),
        "",
        "## Key Insights & Findings",
        "",
        "- The strongest predictive Signals come from the signed driver analysis and SHAP outputs.",
        "- Segmentation reveals recurring audience behavior groups that can be used in targeted moderation.",
        "- Anomaly detection helps identify extreme or suspicious content that deserves analyst attention.",
        "",
        "## Forecasting Insights",
        "",
        "Credibility-score forecasts and related trend plots are available in the reports and assets folders for stakeholder review.",
        "",
        "## Visualization Gallery",
        "",
        "![Forecast Credibility](outputs/plots/prophet/forecast_credibility_score.png)",
        "![Feature Importance](outputs/plots/xgboost/xgboost_feature_importance_credibility_score.png)",
        "![Forecast Misinformation](outputs/plots/random_forest/forecast_misinformation_probability.png)",
        "",
        "## How to Interpret Results",
        "",
        "Use the model metrics for fit, the feature importance charts for driver interpretation, and the forecasting and clustering outputs for business planning.",
        "",
    ]
    save_text(paths.reports / "README.md", "\n".join(readme_lines) + "\n")


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    random_state: int = 42,
    xgboost_device: str = "auto",
    resume: bool = True,
    reuse_models: bool = True,
) -> None:
    """Execute the full analytics workflow from one command."""

    paths = create_output_paths(output_dir)
    checkpoint = load_checkpoint(paths) if resume else {"completed_stages": [], "artifacts": {}}
    configure_logging(paths.reports, append=resume and bool(checkpoint.get("completed_stages")))
    logging.info("Advanced analytics pipeline started (resume=%s, reuse_models=%s)", resume, reuse_models)
    timer = StageTimer()

    resolved_device = resolve_xgboost_device(xgboost_device)
    with timer.stage("environment_setup"):
        log_training_environment(resolved_device, paths)

    data_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0

    with timer.stage("data_loading"):
        data = load_dataset(data_path)
    save_checkpoint(paths, "data_loading")

    with timer.stage("feature_engineering"):
        data = add_engineered_features(data)
    save_checkpoint(paths, "feature_engineering")

    model_results: dict[str, ModelResult] = {}
    with timer.stage("model_training"):
        logging.info("XGBoost training device: %s (requested=%s)", resolved_device, xgboost_device)
        model_results = train_xgboost_models(
            data,
            paths,
            random_state,
            resolved_device,
            checkpoint=checkpoint,
            data_mtime=data_mtime,
            reuse_models=reuse_models,
        )
    save_checkpoint(paths, "model_training")

    with timer.stage("shap_explainability"):
        if is_stage_complete(checkpoint, "shap") and (get_model_output_dir(paths, "xgboost", "reports") / "shap_analysis.txt").exists():
            logging.info("Skipping SHAP; checkpoint indicates prior completion")
        else:
            try:
                run_shap_analysis(model_results, paths)
                save_checkpoint(paths, "shap")
            except Exception as exc:
                logging.exception("SHAP explainability failed: %s", exc)

    scaled_cache: ScaledMatrixCache | None = None
    with timer.stage("preprocessing"):
        scaled_cache = prepare_scaled_matrix(data, CLUSTER_FEATURES)
    save_checkpoint(paths, "preprocessing")

    clustered: pd.DataFrame | None = None
    with timer.stage("clustering_training"):
        clustered_dataset_path = get_dataset_output_dir(paths, "clustering") / "clustered_data.csv"
        if is_stage_complete(checkpoint, "clustering") and clustered_dataset_path.exists():
            logging.info("Loading clustered dataset from checkpoint")
            clustered = pd.read_csv(clustered_dataset_path)
        else:
            try:
                clustered = run_clustering(data, paths, random_state, scaled_cache=scaled_cache)
                save_checkpoint(paths, "clustering", {"clustered_data": str(clustered_dataset_path)})
            except Exception as exc:
                logging.exception("Clustering failed: %s", exc)
                if clustered_dataset_path.exists():
                    clustered = pd.read_csv(clustered_dataset_path)

    anomalies: pd.DataFrame | None = None
    with timer.stage("anomaly_detection"):
        anomalies_dataset_path = get_dataset_output_dir(paths, "anomaly_detection") / "anomalies.csv"
        if is_stage_complete(checkpoint, "anomaly_detection") and anomalies_dataset_path.exists():
            logging.info("Loading anomaly dataset from checkpoint")
            anomalies = pd.read_csv(anomalies_dataset_path)
        else:
            try:
                scored = run_anomaly_detection(data, paths, random_state, scaled_cache=scaled_cache)
                anomalies = scored[scored["anomaly_label"] == -1] if "anomaly_label" in scored.columns else scored
                save_checkpoint(paths, "anomaly_detection", {"anomalies": str(anomalies_dataset_path)})
            except Exception as exc:
                logging.exception("Anomaly detection failed: %s", exc)
                if anomalies_dataset_path.exists():
                    anomalies = pd.read_csv(anomalies_dataset_path)

    forecasts: dict[str, pd.DataFrame] = {}
    with timer.stage("forecasting"):
        daily_metrics_path = get_dataset_output_dir(paths, "forecasting") / "daily_metrics.csv"
        if is_stage_complete(checkpoint, "forecasting") and daily_metrics_path.exists():
            logging.info("Loading forecasts from checkpoint")
            for metric in ["engagement_velocity", "misinformation_probability", "post_volume"]:
                forecast_path = get_dataset_output_dir(paths, "forecasting") / f"forecast_{metric}.csv"
                if forecast_path.exists():
                    forecasts[metric] = pd.read_csv(forecast_path, parse_dates=["analysis_date"])
        else:
            try:
                forecasts = run_forecasting(data, paths)
                save_checkpoint(paths, "forecasting")
            except Exception as exc:
                logging.exception("Forecasting failed: %s", exc)

    with timer.stage("executive_summary"):
        if clustered is not None and anomalies is not None:
            try:
                write_executive_summary(model_results, clustered, anomalies, forecasts, paths)
                save_checkpoint(paths, "executive_summary")
            except Exception as exc:
                logging.exception("Executive summary failed: %s", exc)

    with timer.stage("insights_reporting"):
        try:
            clustering_comparison = None
            kmeans_reports_dir = get_model_output_dir(paths, "kmeans", "reports")
            isolation_reports_dir = get_model_output_dir(paths, "isolation_forest", "reports")
            if (kmeans_reports_dir / "clustering_model_comparison.csv").exists():
                clustering_comparison = pd.read_csv(kmeans_reports_dir / "clustering_model_comparison.csv")
            anomaly_metrics = None
            if (isolation_reports_dir / "anomaly_metrics.csv").exists():
                anomaly_metrics = pd.read_csv(isolation_reports_dir / "anomaly_metrics.csv")
            supervised_anomaly_metrics = None
            if (isolation_reports_dir / "anomaly_supervised_metrics.csv").exists():
                supervised_anomaly_metrics = pd.read_csv(isolation_reports_dir / "anomaly_supervised_metrics.csv")
            write_project_insights_report(
                output_dir,
                paths,
                model_results,
                clustered,
                anomalies,
                forecasts,
                clustering_comparison=clustering_comparison,
                anomaly_metrics=anomaly_metrics,
                supervised_anomaly_metrics=supervised_anomaly_metrics,
            )
            save_checkpoint(paths, "insights_reporting")
        except Exception as exc:
            logging.exception("Insights reporting failed: %s", exc)

    timer.write_report(paths.reports / "pipeline_timings.txt")
    logging.info("Pipeline completed. Outputs saved to %s", paths.root)
    logging.info("Stage timings: %s", timer.timings)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Run the advanced crisis analytics pipeline.")
    parser.add_argument("--data", type=Path, default=Path("../Medalian/Silver.csv"), help="Path to Silver.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Project/output base directory")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility seed")
    parser.add_argument(
        "--xgboost-device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Use cuda for GPU training by default, auto to detect CUDA, or cpu for fallback runs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from pipeline_checkpoint.json when stages already completed.",
    )
    parser.add_argument(
        "--reuse-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse saved model pickles when they are newer than the input dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        random_state=args.random_state,
        xgboost_device=args.xgboost_device,
        resume=args.resume,
        reuse_models=args.reuse_models,
    )
