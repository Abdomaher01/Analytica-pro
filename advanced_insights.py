"""Advanced analytics pipeline for the social media crisis dataset.
The script reads Silver.csv, creates the outputs folder structure, trains
models, generates reports, and saves publication-ready plots and datasets.
"""
# Importing modules
from __future__ import annotations
import argparse
import logging
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
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
    "engagement_velocity",
    "misinformation_probability",
    "credibility_score",
]

# These are the three supervised learning questions the project answers.
# Keeping them in one list lets the same training/evaluation code run for each
# target instead of copying three nearly identical modeling blocks.
TARGET_LABELS = {
    "engagement_velocity": "Engagement Velocity",
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


@dataclass
class OutputPaths:
    """Centralized output paths so the project is easy to move."""

    root: Path
    models: Path
    plots: Path
    reports: Path
    datasets: Path


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

# Status Logging in terminal
def configure_logging(report_dir: Path) -> None:
    """Log progress to the console and to outputs/reports/pipeline.log."""

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
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
    )
    for path in [paths.root, paths.models, paths.plots, paths.reports, paths.datasets]:
        path.mkdir(parents=True, exist_ok=True)
    return paths

# Save reports
def save_text(path: Path, content: str) -> None:
    """Write UTF-8 text reports."""

    path.write_text(content.strip() + "\n", encoding="utf-8")


def save_pickle(path: Path, obj: Any) -> None:
    """Persist Python objects without adding non-requested dependencies."""

    with path.open("wb") as file:
        pickle.dump(obj, file)


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
    month_number = data["month"].map(month_lookup).fillna(1).astype(int)
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
    if device == "cpu":
        return {"tree_method": "hist"}

    try:
        import xgboost as xgb
        major_version = int(str(xgb.__version__).split(".")[0])
    except Exception:
        major_version = 2
    if major_version >= 2:
        return {
            "tree_method": "hist",
            "device": "cuda",
        }

    return {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
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
        n_jobs=-1,
        eval_metric="rmse",
        **get_xgboost_gpu_params(device),
    )


def tune_xgboost_model(
    features: pd.DataFrame,
    target_values: pd.Series,
    random_state: int,
    device: str,
) -> GridSearchCV:
    """Tune an XGBoost pipeline with cross-validation."""

    # The Pipeline keeps preprocessing and modeling together. This is important:
    # during cross-validation, preprocessing is learned only from each training
    # fold, which avoids accidentally learning from validation data.
    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(features)),
            ("model", make_xgb_regressor(random_state, device)),
        ]
    )

    # The grid is intentionally small enough for a laptop project while still
    # testing the main XGBoost tradeoffs: tree count, depth, learning rate, and
    # regularization.
    param_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.03, 0.08],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1.0, 3.0],
    }
    folds = KFold(n_splits=3, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=folds,
        # GPU training should not launch many parallel fits against one laptop GPU.
        n_jobs=1 if device == "cuda" else -1,
        verbose=0,
    )
    search.fit(features, target_values)
    return search


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


def estimate_signed_effects(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Estimate positive and negative drivers with permutation direction.

    Tree importance is unsigned. To explain directional effects, this function
    measures each raw feature's correlation with predictions and multiplies it
    by permutation importance. Positive scores suggest the feature increases the
    target; negative scores suggest the opposite.
    """

    # Permutation importance answers: "How much worse does the model get if this
    # feature is shuffled?" It works on the full pipeline, so it measures raw
    # input columns instead of only post-encoding feature names.
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
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
    plt.savefig(paths.plots / f"xgboost_feature_importance_{target}.png", dpi=200)
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
    plt.savefig(paths.plots / f"xgboost_actual_vs_predicted_{target}.png", dpi=200)
    plt.close()


def train_xgboost_models(
    data: pd.DataFrame,
    paths: OutputPaths,
    random_state: int,
    device: str,
) -> dict[str, ModelResult]:
    """Train, tune, evaluate, and save one XGBoost model per target."""

    logging.info("Starting XGBoost insight engine")
    results: dict[str, ModelResult] = {}

    for target in TARGETS:
        logging.info("Training XGBoost model for %s", target)
        features, target_values = get_feature_target_split(data, target)
        # A fixed random_state makes the train/test split reproducible for demos,
        # reports, and grading. The test set is held back until final evaluation.
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_values,
            test_size=0.2,
            random_state=random_state,
        )

        search = tune_xgboost_model(x_train, y_train, random_state, device)
        best_model = search.best_estimator_
        predictions = best_model.predict(x_test)
        metrics = evaluate_regression(y_test, predictions)
        metrics["best_cv_rmse"] = abs(search.best_score_)

        importance = get_feature_importance(best_model)
        signed_effects = estimate_signed_effects(best_model, x_test, y_test)

        model_token = safe_filename(target)
        save_pickle(paths.models / f"xgboost_{model_token}.pkl", best_model)
        pd.DataFrame([metrics]).to_csv(paths.reports / f"xgboost_metrics_{model_token}.csv", index=False)
        importance.to_csv(paths.reports / f"xgboost_feature_importance_{model_token}.csv", index=False)
        signed_effects.to_csv(paths.reports / f"xgboost_signed_drivers_{model_token}.csv", index=False)
        pd.DataFrame({"actual": y_test, "predicted": predictions}).to_csv(
            paths.datasets / f"xgboost_predictions_{model_token}.csv",
            index=False,
        )

        plot_feature_importance(importance, target, paths)
        plot_prediction_quality(y_test, predictions, target, paths)

        results[target] = ModelResult(
            target=target,
            model=best_model,
            metrics=metrics,
            feature_importance=importance,
            signed_effects=signed_effects,
            x_test_raw=x_test,
            y_test=y_test,
            predictions=predictions,
        )

    write_xgboost_report(results, paths)
    return results


def recommendation_for_target(target: str, positive_driver: str, negative_driver: str) -> str:
    """Create a practical recommendation based on model drivers."""

    if target == "engagement_velocity":
        return (
            f"Prioritize content and response playbooks associated with {positive_driver}, "
            f"while monitoring weak engagement signals linked to {negative_driver}."
        )
    if target == "misinformation_probability":
        return (
            f"Escalate review queues when {positive_driver} rises, and use patterns like "
            f"{negative_driver} as lower-risk benchmarks."
        )
    return (
        f"Amplify credible content patterns connected to {positive_driver}, and audit "
        f"content conditions where {negative_driver} suppresses credibility."
    )


def write_xgboost_report(results: dict[str, ModelResult], paths: OutputPaths) -> None:
    """Generate the main XGBoost insight report."""

    sections = ["XGBoost Insight Engine", "=" * 24]
    for target, result in results.items():
        # Reports convert model artifacts into presentation-ready language:
        # metrics show quality, feature tables show drivers, and recommendations
        # connect those drivers to actions.
        positive = result.signed_effects.head(10)
        negative = result.signed_effects.tail(10).sort_values("directional_effect")
        top_positive = positive.iloc[0]["feature"] if not positive.empty else "the strongest positive driver"
        top_negative = negative.iloc[0]["feature"] if not negative.empty else "the strongest negative driver"

        sections.extend(
            [
                "",
                TARGET_LABELS[target],
                "-" * len(TARGET_LABELS[target]),
                f"MAE: {result.metrics['mae']:.4f}",
                f"RMSE: {result.metrics['rmse']:.4f}",
                f"R2: {result.metrics['r2']:.4f}",
                f"Best cross-validated RMSE: {result.metrics['best_cv_rmse']:.4f}",
                "",
                "Top 20 important features:",
                result.feature_importance.head(20).to_string(index=False),
                "",
                "Strongest positive drivers:",
                positive[["feature", "directional_effect"]].to_string(index=False),
                "",
                "Strongest negative drivers:",
                negative[["feature", "directional_effect"]].to_string(index=False),
                "",
                "Recommendation:",
                recommendation_for_target(target, str(top_positive), str(top_negative)),
            ]
        )

    save_text(paths.reports / "xgboost_insights.txt", "\n".join(sections))


def run_shap_analysis(results: dict[str, ModelResult], paths: OutputPaths) -> dict[str, pd.DataFrame]:
    """Create SHAP plots and a plain-English explanation report."""

    logging.info("Starting SHAP explainability")
    shap_tables: dict[str, pd.DataFrame] = {}
    sections = ["SHAP Explainability Analysis", "=" * 28]

    if shap is None:
        message = f"SHAP is not available in this environment: {SHAP_IMPORT_ERROR}"
        logging.warning(message)
        save_text(paths.reports / "shap_analysis.txt", message)
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
        shap_table.to_csv(paths.reports / f"shap_importance_{target}.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, transformed_df, show=False, max_display=20)
        plt.title(f"SHAP Summary: {TARGET_LABELS[target]}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"shap_summary_{target}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_values, transformed_df, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Bar Importance: {TARGET_LABELS[target]}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"shap_bar_{target}.png", dpi=200, bbox_inches="tight")
        plt.close()

        dependence_feature = shap_table.iloc[0]["feature"]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(dependence_feature, shap_values, transformed_df, show=False)
        plt.title(f"SHAP Dependence: {dependence_feature}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"shap_dependence_{target}.png", dpi=200, bbox_inches="tight")
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

    save_text(paths.reports / "shap_analysis.txt", "\n".join(sections))
    return shap_tables


def prepare_scaled_matrix(data: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Impute and scale numeric features for clustering and anomaly detection."""

    available = [column for column in columns if column in data.columns]
    matrix = data[available].apply(pd.to_numeric, errors="coerce")
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(matrix),
        columns=available,
        index=data.index,
    )
    scaled = StandardScaler().fit_transform(imputed)
    return imputed, scaled


def choose_kmeans_clusters(scaled: np.ndarray, paths: OutputPaths, random_state: int) -> tuple[int, pd.DataFrame]:
    """Select K using silhouette score and save elbow diagnostics."""

    max_k = min(10, len(scaled) - 1)
    candidate_ks = list(range(2, max_k + 1))
    rows = []

    # K-Means needs the number of clusters in advance. We compare candidate K
    # values with inertia and silhouette score, then choose the strongest
    # silhouette score because it rewards separated, compact clusters.
    for k in candidate_ks:
        labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(scaled)
        rows.append(
            {
                "k": k,
                "inertia": KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(scaled).inertia_,
                "silhouette_score": silhouette_score(scaled, labels),
            }
        )

    scores = pd.DataFrame(rows)
    best_k = int(scores.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    scores.to_csv(paths.reports / "kmeans_cluster_selection.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=scores, x="k", y="inertia", marker="o")
    plt.title("K-Means Elbow Method")
    plt.tight_layout()
    plt.savefig(paths.plots / "kmeans_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=scores, x="k", y="silhouette_score", marker="o")
    plt.title("K-Means Silhouette Scores")
    plt.tight_layout()
    plt.savefig(paths.plots / "kmeans_silhouette.png", dpi=200)
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


def run_clustering(data: pd.DataFrame, paths: OutputPaths, random_state: int) -> pd.DataFrame:
    """Run HDBSCAN when available and K-Means with automatic K selection."""

    logging.info("Starting user and post segmentation")
    numeric_data, scaled = prepare_scaled_matrix(data, CLUSTER_FEATURES)
    clustered = data.copy()

    # Clustering is unsupervised: there is no target column. The goal is to find
    # natural behavior groups that can guide moderation or communication strategy.
    best_k, _ = choose_kmeans_clusters(scaled, paths, random_state)
    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=random_state)
    clustered["kmeans_cluster"] = kmeans.fit_predict(scaled)
    save_pickle(paths.models / "kmeans_segmentation.pkl", kmeans)

    if HDBSCAN is not None:
        hdbscan_model = HDBSCAN(min_cluster_size=max(10, int(len(data) * 0.02)))
        clustered["hdbscan_cluster"] = hdbscan_model.fit_predict(scaled)
        save_pickle(paths.models / "hdbscan_segmentation.pkl", hdbscan_model)
    else:
        clustered["hdbscan_cluster"] = np.nan

    clustered.to_csv(paths.datasets / "clustered_data.csv", index=False)

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
    summary.to_csv(paths.reports / "clustering_summary.csv", index=False)

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
    plt.savefig(paths.plots / "clusters_engagement_misinformation.png", dpi=200)
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
    save_text(paths.reports / "clustering_report.txt", "\n".join(report_lines))
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


def run_anomaly_detection(data: pd.DataFrame, paths: OutputPaths, random_state: int) -> pd.DataFrame:
    """Detect suspicious accounts, unusual posts, and engagement spikes."""

    logging.info("Starting anomaly detection")
    numeric_data, scaled = prepare_scaled_matrix(data, CLUSTER_FEATURES)
    # IsolationForest works well for rare-pattern detection because it tries to
    # isolate unusual rows quickly across many random trees.
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(scaled)
    scores = model.decision_function(scaled)
    save_pickle(paths.models / "isolation_forest_anomaly_detector.pkl", model)

    scored = data.copy()
    scored["anomaly_label"] = labels
    scored["anomaly_score"] = scores
    thresholds = numeric_data.quantile(0.95)
    scored["anomaly_reason"] = [
        explain_anomaly_reasons(numeric_data.loc[index], thresholds)
        if label == -1
        else "normal behavior"
        for index, label in zip(scored.index, labels)
    ]

    anomalies = scored[scored["anomaly_label"] == -1].sort_values("anomaly_score")
    anomalies.to_csv(paths.datasets / "anomalies.csv", index=False)
    pd.DataFrame({"anomaly_count": [len(anomalies)], "total_records": [len(scored)]}).to_csv(
        paths.reports / "anomaly_metrics.csv",
        index=False,
    )

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
    plt.savefig(paths.plots / "anomalies_engagement_misinformation.png", dpi=200)
    plt.close()

    top_reasons = anomalies["anomaly_reason"].value_counts().head(10)
    report = [
        "Anomaly Detection Report",
        "========================",
        f"Flagged records: {len(anomalies):,} of {len(scored):,}",
        "",
        "Most common anomaly reasons:",
        top_reasons.to_string(),
        "",
        "Records are flagged by IsolationForest when their combined account, engagement, credibility, toxicity, and misinformation profile is rare compared with the rest of the dataset.",
    ]
    save_text(paths.reports / "anomaly_report.txt", "\n".join(report))
    return anomalies


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
    series["time_index"] = np.arange(len(series))
    # This fallback is not a full time-series model; it learns a simple nonlinear
    # trend from time index to value so the project still runs without Prophet.
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(series[["time_index"]], series["value"])

    future_index = np.arange(len(series) + periods)
    future_dates = pd.date_range(series["analysis_date"].min(), periods=len(series) + periods, freq="D")
    predictions = model.predict(pd.DataFrame({"time_index": future_index}))
    residual = series["value"] - model.predict(series[["time_index"]])
    interval = 1.96 * residual.std(ddof=0)

    return pd.DataFrame(
        {
            "analysis_date": future_dates,
            "yhat": predictions,
            "yhat_lower": predictions - interval,
            "yhat_upper": predictions + interval,
        }
    )


def run_forecasting(data: pd.DataFrame, paths: OutputPaths, periods: int = 30) -> dict[str, pd.DataFrame]:
    """Forecast engagement, misinformation, and post volume for 30 days."""

    logging.info("Starting time-series forecasting")
    daily = aggregate_daily_metrics(data)
    daily.to_csv(paths.datasets / "daily_metrics.csv", index=False)

    forecasts = {}
    report_lines = [
        "Forecast Report",
        "===============",
        f"Forecast horizon: {periods} days",
        f"Method: {'Prophet' if Prophet is not None else 'RandomForest trend fallback'}",
        "",
    ]

    for metric in ["engagement_velocity", "misinformation_probability", "post_volume"]:
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
        forecast.to_csv(paths.datasets / f"forecast_{metric}.csv", index=False)

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=series, x="analysis_date", y="value", label="actual")
        sns.lineplot(data=forecast, x="analysis_date", y="yhat", label="forecast")
        plt.fill_between(
            forecast["analysis_date"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            alpha=0.18,
        )
        plt.title(f"30-Day Forecast: {metric.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"forecast_{metric}.png", dpi=200)
        plt.close()

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

    save_text(paths.reports / "forecast_report.txt", "\n".join(report_lines))
    return forecasts


def build_propagation_graph(data: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from parent posts to child posts."""

    graph = nx.DiGraph()
    for _, row in data.iterrows():
        post_id = str(row["post_id"])
        parent = row.get("parent_post_id")
        graph.add_node(post_id)
        # An edge means information flowed from a parent post to a child/share.
        # Removing empty parent values keeps standalone posts as isolated nodes.
        if pd.notna(parent) and str(parent) not in {"0", "0.0", "", "nan", "None"}:
            graph.add_edge(str(parent), post_id)
    return graph


def run_network_analysis(data: pd.DataFrame, paths: OutputPaths) -> dict[str, Any]:
    """Analyze propagation graph centrality and cascade structure."""

    logging.info("Starting social network analysis")
    graph = build_propagation_graph(data)
    undirected = graph.to_undirected()

    # Centrality metrics identify posts that are structurally important in the
    # propagation network, not just posts with high raw engagement.
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=min(500, len(graph)) if len(graph) > 500 else None, seed=42)
    closeness = nx.closeness_centrality(graph)
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)

    centrality = (
        pd.DataFrame(
            {
                "node": list(graph.nodes()),
                "degree_centrality": [degree.get(node, 0) for node in graph.nodes()],
                "betweenness_centrality": [betweenness.get(node, 0) for node in graph.nodes()],
                "closeness_centrality": [closeness.get(node, 0) for node in graph.nodes()],
            }
        )
        .sort_values(["degree_centrality", "betweenness_centrality"], ascending=False)
        .reset_index(drop=True)
    )
    centrality.to_csv(paths.reports / "network_centrality.csv", index=False)

    largest_nodes = components[0] if components else []
    subgraph_nodes = list(largest_nodes)[: min(150, len(largest_nodes))]
    subgraph = graph.subgraph(subgraph_nodes)

    plt.figure(figsize=(12, 9))
    if len(subgraph) > 0:
        layout = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, layout, node_size=35, node_color="#2f6f9f", alpha=0.75)
        nx.draw_networkx_edges(subgraph, layout, arrows=False, alpha=0.25)
    plt.title("Largest Propagation Cascade")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(paths.plots / "network_largest_cascade.png", dpi=200)
    plt.close()

    report = [
        "Social Network Analysis",
        "=======================",
        f"Nodes: {graph.number_of_nodes():,}",
        f"Edges: {graph.number_of_edges():,}",
        f"Connected components: {len(components):,}",
        f"Largest cascade size: {len(largest_nodes):,}",
        "",
        "Most influential nodes by centrality:",
        centrality.head(20).to_string(index=False),
    ]
    save_text(paths.reports / "network_analysis.txt", "\n".join(report))

    return {
        "graph": graph,
        "centrality": centrality,
        "largest_cascade_size": len(largest_nodes),
        "component_count": len(components),
    }


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
    network_result: dict[str, Any],
    paths: OutputPaths,
) -> None:
    """Generate a stakeholder-ready business summary."""

    # This final report translates technical outputs into decision questions.
    # It is meant for slides or stakeholder discussion, not model debugging.
    cluster_count = clustered["kmeans_cluster"].nunique()
    anomaly_count = len(anomalies)
    engagement_driver = summarize_top_driver(model_results, "engagement_velocity")
    misinformation_driver = summarize_top_driver(model_results, "misinformation_probability")
    credibility_driver = summarize_top_driver(model_results, "credibility_score")

    def forecast_direction(metric: str) -> str:
        forecast = forecasts.get(metric)
        if forecast is None or forecast.empty:
            return "stable"
        first = forecast.tail(30)["yhat"].iloc[0]
        last = forecast.tail(30)["yhat"].iloc[-1]
        return "upward" if last > first else "downward"

    summary = f"""
Executive Summary
=================

1. What drives virality?
Virality is most strongly associated with {engagement_driver}. Content teams should monitor this signal because it is the clearest modeled driver of engagement velocity.

2. What drives misinformation?
Misinformation risk is most strongly associated with {misinformation_driver}. This signal should be prioritized in moderation queues and early-warning dashboards.

3. What characterizes credible content?
Credibility is most strongly associated with {credibility_driver}. Stakeholders should amplify content patterns that match this profile and audit posts that move in the opposite direction.

4. Which user groups exist?
The segmentation model identified {cluster_count} K-Means user/post groups. These groups separate audiences by engagement, credibility, toxicity, and misinformation risk, enabling differentiated response strategies.

5. Which anomalies were discovered?
IsolationForest flagged {anomaly_count:,} unusual records. These records show rare combinations of audience size, engagement, toxicity, credibility, and misinformation risk, making them strong candidates for human review.

6. What trends are expected next month?
The forecast suggests an {forecast_direction('engagement_velocity')} engagement trend, an {forecast_direction('misinformation_probability')} misinformation trend, and an {forecast_direction('post_volume')} post-volume trend over the next 30 days.

7. What are the most important strategic recommendations?
- Build monitoring dashboards around the strongest engagement and misinformation drivers.
- Prioritize anomalous high-risk records for analyst review.
- Use cluster profiles to tailor crisis response messages by audience behavior.
- Track cascade centrality because the largest cascade contains {network_result.get('largest_cascade_size', 0):,} nodes and can reveal influential propagation points.
- Re-run this pipeline after each major crisis phase to keep forecasts and model explanations current.
"""
    save_text(paths.reports / "executive_summary.txt", summary)


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    random_state: int = 42,
    xgboost_device: str = "cuda",
) -> None:
    """Execute the full analytics workflow from one command."""

    paths = create_output_paths(output_dir)
    configure_logging(paths.reports)
    logging.info("Advanced analytics pipeline started")

    # The order matters: clean/engineer features first, then supervised models,
    # then explainability and unsupervised analyses, then the executive summary
    # that depends on all previous artifacts.
    data = load_dataset(data_path)
    data = add_engineered_features(data)

    logging.info("XGBoost training device: %s", xgboost_device)
    model_results = train_xgboost_models(data, paths, random_state, xgboost_device)
    run_shap_analysis(model_results, paths)
    clustered = run_clustering(data, paths, random_state)
    anomalies = run_anomaly_detection(data, paths, random_state)
    forecasts = run_forecasting(data, paths)
    network_result = run_network_analysis(data, paths)
    write_executive_summary(model_results, clustered, anomalies, forecasts, network_result, paths)

    logging.info("Pipeline completed successfully. Outputs saved to %s", paths.root)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Run the advanced crisis analytics pipeline.")
    parser.add_argument("--data", type=Path, default=Path("Silver.csv"), help="Path to Silver.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Project/output base directory")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility seed")
    parser.add_argument(
        "--xgboost-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Use cuda for RTX/NVIDIA GPU training or cpu for fallback runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        random_state=args.random_state,
        xgboost_device=args.xgboost_device,
    )
