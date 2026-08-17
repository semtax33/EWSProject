"""Purged walk-forward models for the separate continuous-return research track."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


REGRESSION_MODELS = ("elastic_net", "linear_svr", "rbf_svr")


def make_regression_model(model_type, random_seed=42):
    """Return a small, pre-declared model; no holdout hyperparameter search."""
    if model_type == "elastic_net":
        estimator = ElasticNet(
            alpha=0.05,
            l1_ratio=0.5,
            max_iter=20_000,
            random_state=random_seed,
        )
    elif model_type == "linear_svr":
        estimator = LinearSVR(
            C=0.25,
            epsilon=0.05,
            max_iter=20_000,
            dual="auto",
            random_state=random_seed,
        )
    elif model_type == "rbf_svr":
        estimator = SVR(C=1.0, epsilon=0.05, gamma="scale", kernel="rbf")
    else:
        raise ValueError(f"Unknown regression model: {model_type}")
    return Pipeline([("feature_scaler", StandardScaler()), ("model", estimator)])


def walk_forward_regression(
    X,
    target,
    *,
    eval_start,
    eval_end,
    model_type,
    min_train=84,
    purge=3,
    refit_every=1,
    winsor_limits=(0.01, 0.99),
    random_seed=42,
):
    """Predict returns with train-only target winsorization and scaling."""
    X = X.sort_index()
    target = target.reindex(X.index)
    eval_dates = X.index[
        (X.index >= pd.Timestamp(eval_start)) & (X.index <= pd.Timestamp(eval_end))
    ]
    prediction = pd.Series(np.nan, index=eval_dates, name=model_type, dtype=float)
    audit_rows = []
    fitted = None
    target_mean = np.nan
    target_scale = np.nan

    for position, test_date in enumerate(eval_dates):
        test_x = X.loc[[test_date]]
        if test_x.isna().any(axis=None):
            continue
        if fitted is None or position % refit_every == 0:
            cutoff = test_date.to_period("M") - purge
            train_mask = X.index.to_period("M") <= cutoff
            train = pd.concat(
                [X.loc[train_mask], target.loc[train_mask].rename("target")], axis=1
            ).dropna()
            if len(train) < min_train:
                continue
            lower, upper = train["target"].quantile(list(winsor_limits))
            winsorized = train["target"].clip(lower=lower, upper=upper)
            target_mean = float(winsorized.mean())
            target_scale = float(winsorized.std(ddof=0))
            if not np.isfinite(target_scale) or target_scale <= 0:
                continue
            scaled_target = (winsorized - target_mean) / target_scale
            fitted = make_regression_model(model_type, random_seed=random_seed)
            fitted.fit(train[X.columns], scaled_target)
            audit_rows.append(
                {
                    "prediction_date": test_date,
                    "train_end": train.index.max(),
                    "train_n": len(train),
                    "target_winsor_lower": float(lower),
                    "target_winsor_upper": float(upper),
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                }
            )
        if fitted is not None:
            scaled_prediction = float(fitted.predict(test_x)[0])
            prediction.loc[test_date] = target_mean + target_scale * scaled_prediction

    return prediction, pd.DataFrame(audit_rows)


def regression_metrics(prediction, actual):
    data = pd.concat(
        [prediction.rename("prediction"), actual.rename("actual")], axis=1
    ).dropna()
    if data.empty:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "rank_ic": np.nan,
            "pearson_ic": np.nan,
            "direction_hit_rate": np.nan,
        }
    error = data["prediction"] - data["actual"]
    rank_ic = (
        spearmanr(data["prediction"], data["actual"]).statistic
        if data["prediction"].nunique() > 1 and data["actual"].nunique() > 1
        else np.nan
    )
    return {
        "n": len(data),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
        "pearson_ic": float(data["prediction"].corr(data["actual"])),
        "direction_hit_rate": float(
            (np.sign(data["prediction"]) == np.sign(data["actual"])).mean()
        ),
    }


def select_regression_model(metrics):
    """Freeze using pre-holdout rank IC; RMSE breaks ties/fallbacks."""
    eligible = metrics.loc[(metrics["n"] >= 36) & metrics["rank_ic"].notna()].copy()
    if eligible.empty:
        raise ValueError("No regression model has enough pre-holdout OOS predictions")
    positive = eligible.loc[eligible["rank_ic"] > 0]
    pool = positive if not positive.empty else eligible
    selected = pool.sort_values(
        ["rank_ic", "rmse", "model"], ascending=[False, True, True]
    ).iloc[0]
    return str(selected["model"]), (
        "highest positive pre-holdout outer-OOS Rank IC; RMSE tie-break"
        if not positive.empty
        else "no positive Rank IC; least-negative Rank IC fallback, RMSE tie-break"
    )
