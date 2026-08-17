import numpy as np
import pandas as pd
from itertools import combinations

from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import rankdata

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
)


# ============================================================
# TARGET
# ============================================================

def build_target(
    market_price,
    horizon=3,
    threshold=0.0,
):

    future_return = (
        market_price.shift(-horizon)
        / market_price
        - 1
    )

    y = pd.Series(
        np.nan,
        index=market_price.index,
        dtype=float,
        name="y",
    )

    valid = future_return.notna()

    y.loc[valid] = (
        future_return.loc[valid]
        > threshold
    ).astype(float)

    return pd.DataFrame({
        "future_return": future_return,
        "y": y,
    })


def build_future_drawdown_target(
    market_price,
    horizon=3,
    drawdown_threshold=-0.05,
):
    """Label whether the forward monthly path avoids a declared drawdown.

    The label is one when every month-end return from the observation price
    through ``horizon`` stays above ``drawdown_threshold``.  It is computed
    only from the price path and is therefore suitable as a model label; the
    usual purged walk-forward split still controls when that label becomes
    available for training.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not -1.0 < drawdown_threshold < 0.0:
        raise ValueError("drawdown_threshold must be strictly between -1 and 0")
    price = pd.to_numeric(market_price, errors="coerce").astype(float)
    forward_path = pd.concat(
        [price.shift(-step) / price - 1.0 for step in range(1, horizon + 1)],
        axis=1,
    )
    future_return = price.shift(-horizon) / price - 1.0
    future_path_drawdown = forward_path.min(axis=1)
    complete_path = forward_path.notna().all(axis=1)
    y = pd.Series(np.nan, index=price.index, dtype=float, name="y")
    y.loc[complete_path] = (
        future_path_drawdown.loc[complete_path] > drawdown_threshold
    ).astype(float)
    return pd.DataFrame(
        {
            "future_return": future_return,
            "future_path_drawdown": future_path_drawdown.where(complete_path),
            "y": y,
        }
    )


def build_cash_excess_target(
    market_price,
    cash_yield,
    *,
    horizon=3,
):
    """Label whether the forward index return beats the current cash hurdle.

    ``cash_yield`` is the annualized percentage yield observable on the signal
    date.  The label therefore does not require a future cash-rate path and is
    fully determined once the horizon-end market price is observed.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    price = pd.to_numeric(market_price, errors="coerce").astype(float)
    annual_yield = pd.to_numeric(cash_yield, errors="coerce").reindex(price.index)
    future_return = price.shift(-horizon) / price - 1.0
    cash_hurdle = (
        (1.0 + annual_yield.clip(lower=-99.0) / 100.0) ** (horizon / 12.0)
        - 1.0
    )
    valid = future_return.notna() & cash_hurdle.notna()
    y = pd.Series(np.nan, index=price.index, dtype=float, name="y")
    y.loc[valid] = (
        future_return.loc[valid] > cash_hurdle.loc[valid]
    ).astype(float)
    return pd.DataFrame(
        {
            "future_return": future_return,
            "cash_hurdle": cash_hurdle,
            "y": y,
        }
    )


def build_model_target(
    market_price,
    *,
    mode="absolute_positive",
    horizon=3,
    return_threshold=0.0,
    drawdown_threshold=-0.05,
    cash_yield=None,
):
    """Build a named, auditable classification target."""
    if mode == "absolute_positive":
        return build_target(
            market_price,
            horizon=horizon,
            threshold=return_threshold,
        )
    if mode == "future_drawdown":
        return build_future_drawdown_target(
            market_price,
            horizon=horizon,
            drawdown_threshold=drawdown_threshold,
        )
    if mode == "cash_excess":
        if cash_yield is None:
            raise ValueError("cash_yield is required for cash_excess target")
        return build_cash_excess_target(
            market_price,
            cash_yield,
            horizon=horizon,
        )
    raise ValueError(f"Unknown model target mode: {mode}")


# ============================================================
# MODEL
# ============================================================


class LinearBackboneMLPRiskVeto(BaseEstimator, ClassifierMixin):
    """Small-sample neural classifier with a one-sided defensive veto.

    The regularized Logistic model provides the stable directional backbone.
    The MLP is allowed to reduce an aggressive risk-on probability to neutral
    when it disagrees, but it can never create extra equity exposure.  This is
    useful when the nonlinear learner has signal but too little data to support
    autonomous portfolio sizing.
    """

    def __init__(
        self,
        mlp_params=None,
        *,
        risk_on_threshold=0.65,
        mlp_veto_threshold=0.50,
        neutral_probability=0.50,
        linear_c=0.5,
        random_state=42,
    ):
        self.mlp_params = mlp_params
        self.risk_on_threshold = risk_on_threshold
        self.mlp_veto_threshold = mlp_veto_threshold
        self.neutral_probability = neutral_probability
        self.linear_c = linear_c
        self.random_state = random_state

    @staticmethod
    def _positive_probability(model, X):
        probabilities = model.predict_proba(X)
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        return probabilities[:, positive_index]

    def fit(self, X, y, sample_weight=None):
        self.linear_ = LogisticRegression(
            C=self.linear_c,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=2000,
        )
        params = dict(self.mlp_params or {})
        self.mlp_ = MLPClassifier(
            **params,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.linear_.fit(X, y, sample_weight=sample_weight)
        if sample_weight is None:
            self.mlp_.fit(X, y)
        else:
            self.mlp_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.linear_.classes_
        self.n_features_in_ = self.linear_.n_features_in_
        return self

    def predict_proba(self, X):
        linear_probability = self._positive_probability(self.linear_, X)
        mlp_probability = self._positive_probability(self.mlp_, X)
        probability = linear_probability.copy()
        veto = (
            (linear_probability >= self.risk_on_threshold)
            & (mlp_probability < self.mlp_veto_threshold)
        )
        probability[veto] = self.neutral_probability
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(self.classes_.dtype)


def _make_time_series_calibration_splits(
    y,
    n_splits=3,
    gap=3,
):
    """
    SVM probability calibration용 시계열 CV.

    train → gap → calibration
    순서를 강제한다.
    """

    y_array = np.asarray(y)

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
    )

    dummy_x = np.zeros(
        (len(y_array), 1)
    )

    valid_splits = []

    for train_idx, valid_idx in splitter.split(dummy_x):

        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]

        # 양쪽에 0/1 class가 모두 있어야
        # probability calibration 가능
        if np.unique(y_train).size < 2:
            continue

        if np.unique(y_valid).size < 2:
            continue

        valid_splits.append(
            (train_idx, valid_idx)
        )

    if len(valid_splits) < 2:
        raise ValueError(
            "SVM calibration에 필요한 "
            "시계열 fold가 부족해."
        )

    return valid_splits

def make_model(
    model_type="logistic",
    horizon=3,
    y_train=None,
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
):

    # ========================================================
    # LOGISTIC
    # ========================================================

    if model_type == "logistic":

        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2000,
                )
            ),
        ])

    # ========================================================
    # LOW-COMPLEXITY NONLINEAR MODEL FOR UNIVARIATE SCREENING
    # ========================================================

    elif model_type == "spline_logistic":

        return Pipeline([
            (
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=2,
                    knots="quantile",
                    extrapolation="linear",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ])


    # ========================================================
    # SVM
    # ========================================================

    elif model_type in {"svm", "svm_rank"}:

        if y_train is None:
            raise ValueError(
                "SVM calibration을 위해 "
                "y_train이 필요해."
            )

        params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
        }

        if svm_params is not None:
            params.update(
                svm_params
            )

        # ----------------------------------------------------
        # StandardScaler + SVM
        #
        # probability=True는 사용 안 한다.
        # ----------------------------------------------------

        base_svm = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "svm",
                SVC(
                    C=params["C"],
                    kernel=params["kernel"],
                    gamma=params["gamma"],
                    cache_size=500,
                )
            ),
        ])

        if model_type == "svm_rank":
            return base_svm

        calibration_cv = (
            _make_time_series_calibration_splits(
                y_train,
                n_splits=calibration_splits,
                gap=horizon,
            )
        )

        # SVM score를 0~1 probability로 변환
        model = CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=calibration_cv,
            ensemble=True,
            n_jobs=1,
        )

        return model

    # ========================================================
    # LOW-COMPLEXITY MULTI-LAYER PERCEPTRON
    # ========================================================

    elif model_type == "mlp":

        params = {
            "hidden_layer_sizes": (8, 4),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.05,
            "max_iter": 500,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 30,
        }
        if mlp_params is not None:
            params.update(mlp_params)

        balance_classes = bool(params.pop("balance_classes", False))
        hybrid_mode = params.pop("hybrid_mode", "autonomous")
        risk_on_threshold = float(params.pop("risk_on_threshold", 0.65))
        mlp_veto_threshold = float(params.pop("mlp_veto_threshold", 0.50))
        neutral_probability = float(params.pop("neutral_probability", 0.50))
        if hybrid_mode not in {"autonomous", "risk_veto"}:
            raise ValueError(f"Unknown MLP hybrid_mode: {hybrid_mode}")
        if hybrid_mode == "risk_veto":
            classifier = LinearBackboneMLPRiskVeto(
                mlp_params=params,
                risk_on_threshold=risk_on_threshold,
                mlp_veto_threshold=mlp_veto_threshold,
                neutral_probability=neutral_probability,
                random_state=random_state,
            )
        else:
            classifier = MLPClassifier(
                **params,
                random_state=random_state,
                early_stopping=False,
            )
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", classifier),
        ])
        model._ews_balance_classes = balance_classes
        return model


    else:
        raise ValueError(
            f"Unknown model_type: {model_type}"
        )


def fit_classification_model(model, X, y):
    """Fit with optional train-only class balancing for an MLP pipeline."""
    if getattr(model, "_ews_balance_classes", False):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return model.fit(X, y, mlp__sample_weight=sample_weight)
    return model.fit(X, y)

# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_predict(
    X,
    y,
    eval_start=None,
    eval_end=None,
    min_train=60,
    purge=3,
    refit_every=1,
    model_type="logistic",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
    max_train=None,
):

    if max_train is not None and max_train < min_train:
        raise ValueError("max_train must be at least min_train")

    X = X.sort_index()

    y = y.reindex(
        X.index
    )

    eval_dates = X.index

    if eval_start is not None:
        eval_dates = eval_dates[
            eval_dates >= pd.Timestamp(
                eval_start
            )
        ]

    if eval_end is not None:
        eval_dates = eval_dates[
            eval_dates <= pd.Timestamp(
                eval_end
            )
        ]

    predictions = pd.Series(
        np.nan,
        index=eval_dates,
        dtype=float,
        name=f"{model_type}_prediction",
    )

    model = None

    for i, test_date in enumerate(
        eval_dates
    ):

        x_test = X.loc[
            [test_date]
        ]

        # 현재 Factor 중 하나라도 없으면
        # 그 달 예측 skip
        if x_test.isna().any(
            axis=None
        ):
            continue

        need_refit = (
            model is None
            or i % refit_every == 0
        )

        if need_refit:

            # =================================================
            # PURGE
            #
            # 3개월 미래수익을 Target으로 쓰면
            # 최근 3개월 label은 현재 시점에서
            # 아직 알 수 없음.
            # =================================================

            cutoff_period = (
                test_date.to_period("M")
                - purge
            )

            train_periods = (
                X.index.to_period("M")
            )

            train_mask = (
                train_periods
                <= cutoff_period
            )

            train = pd.concat(
                [
                    X.loc[train_mask],
                    y.loc[
                        train_mask
                    ].rename("y"),
                ],
                axis=1,
            ).dropna()

            if max_train is not None:
                train = train.tail(max_train)

            if len(train) < min_train:
                continue

            if train["y"].nunique() < 2:
                continue

            try:

                model = make_model(
                    model_type=model_type,
                    horizon=purge,
                    y_train=train["y"],
                    svm_params=svm_params,
                    mlp_params=mlp_params,
                    calibration_splits=(
                        calibration_splits
                    ),
                    random_state=random_state,
                )

                fit_classification_model(
                    model,
                    train[X.columns],
                    train["y"],
                )

            except ValueError:

                # 초기 구간에서 SVM calibration용
                # class가 부족하면 skip
                model = None
                continue

        if model is None:
            continue

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(x_test)[0, 1]
        else:
            # Feature selection optimizes AUC/rank only.  The monotonic
            # sigmoid preserves decision-score ordering without repeating
            # probability calibration for every trial subset.
            decision = float(np.ravel(model.decision_function(x_test))[0])
            probability = 1.0 / (1.0 + np.exp(-np.clip(decision, -35, 35)))

        predictions.loc[
            test_date
        ] = probability

    return predictions


def earliest_walk_forward_prediction_date(
    X,
    y,
    *,
    min_train,
    purge,
):
    """Return when a frozen feature set first has a valid purged fit.

    The date depends only on feature/label availability and declared sample
    requirements, never on prediction quality or future return outcomes.
    """
    if min_train < 1 or purge < 0:
        raise ValueError("min_train and purge are invalid")
    features = X.sort_index()
    labels = y.reindex(features.index)
    complete_features = features.notna().all(axis=1)
    periods = features.index.to_period("M")
    for date in features.index[complete_features]:
        cutoff = date.to_period("M") - purge
        train_mask = (periods <= cutoff) & complete_features.to_numpy()
        train_labels = labels.loc[train_mask].dropna()
        if len(train_labels) < min_train or train_labels.nunique() < 2:
            continue
        return pd.Timestamp(date)
    return None


# ============================================================
# METRICS
# ============================================================

def _auc_or_nan(y, p):

    if len(y) < 10:
        return np.nan

    if y.nunique() < 2:
        return np.nan

    return roc_auc_score(
        y,
        p
    )


def evaluate_probabilities(
    prediction,
    y,
):

    df = pd.concat(
        [
            prediction.rename("p"),
            y.rename("y"),
        ],
        axis=1,
    ).dropna()

    n = len(df)

    if (
        n < 20
        or df["y"].nunique() < 2
    ):
        return {
            "n": n,
            "auc": np.nan,
            "brier": np.nan,
            "accuracy": np.nan,
            "auc_first": np.nan,
            "auc_second": np.nan,
            "stability_gap": np.nan,
            "rank_score": np.nan,
        }

    auc = roc_auc_score(
        df["y"],
        df["p"],
    )

    brier = brier_score_loss(
        df["y"],
        df["p"],
    )

    accuracy = accuracy_score(
        df["y"],
        (
            df["p"] >= 0.5
        ).astype(int),
    )

    midpoint = n // 2

    first = df.iloc[:midpoint]
    second = df.iloc[midpoint:]

    auc_first = _auc_or_nan(
        first["y"],
        first["p"],
    )

    auc_second = _auc_or_nan(
        second["y"],
        second["p"],
    )

    if (
        np.isfinite(auc_first)
        and np.isfinite(auc_second)
    ):

        stability_gap = abs(
            auc_first
            - auc_second
        )

    else:

        stability_gap = 0.10

    # 단순히 최고 AUC만 고르면
    # 한 시기에만 대박 난 factor가 올라옴.
    rank_score = (
        auc
        - 0.25 * stability_gap
    )

    return {
        "n": n,
        "auc": auc,
        "brier": brier,
        "accuracy": accuracy,
        "auc_first": auc_first,
        "auc_second": auc_second,
        "stability_gap": stability_gap,
        "rank_score": rank_score,
    }


# ============================================================
# SINGLE FACTOR SCREEN
# ============================================================

def _screen_one_factor(
    feature,
    X,
    y,
    dev_end,
    eval_start,
    min_train,
    horizon,
    refit_every,
    model_type="logistic",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
):

    x = X[[feature]]

    prediction = walk_forward_predict(
        x,
        y,
        eval_start=eval_start,
        eval_end=dev_end,
        min_train=min_train,
        purge=horizon,
        refit_every=refit_every,
        model_type=model_type,
        svm_params=svm_params,
        mlp_params=mlp_params,
        calibration_splits=calibration_splits,
        random_state=random_state,
    )

    metrics = evaluate_probabilities(
        prediction,
        y,
    )

    return {
        "feature": feature,
        "base": feature.split("__")[0],
        **metrics,
    }


def _fit_fast_univariate_logistic(
    values,
    labels,
    min_train,
    *,
    l2_penalty=2.0,
    max_iter=15,
    batch_size=1024,
):
    """Fit independent one-feature logistic models in vectorized batches.

    This is the high-throughput equivalent of fitting thousands of separate
    standardized univariate logistic regressions.  Missing observations are
    handled per feature, and every feature must independently satisfy the
    training-length and two-class checks.
    """
    n_features = values.shape[1]
    means = np.full(n_features, np.nan, dtype=float)
    scales = np.full(n_features, np.nan, dtype=float)
    intercepts = np.full(n_features, np.nan, dtype=float)
    coefficients = np.full(n_features, np.nan, dtype=float)
    eligible_all = np.zeros(n_features, dtype=bool)
    label_column = labels[:, None]

    for start in range(0, n_features, batch_size):
        stop = min(start + batch_size, n_features)
        block = values[:, start:stop]
        valid = np.isfinite(block)
        counts = valid.sum(axis=0)
        safe_counts = np.maximum(counts, 1)
        safe_block = np.where(valid, block, 0.0)
        mean = safe_block.sum(axis=0) / safe_counts
        centered = np.where(valid, block - mean, 0.0)
        scale = np.sqrt((centered * centered).sum(axis=0) / safe_counts)
        positive = (valid * label_column).sum(axis=0)
        eligible = (
            (counts >= min_train)
            & (positive > 0)
            & (positive < counts)
            & np.isfinite(scale)
            & (scale > 1e-12)
        )
        safe_scale = np.where(eligible, scale, 1.0)
        standardized = centered / safe_scale
        rate = np.clip(positive / safe_counts, 1e-6, 1.0 - 1e-6)
        intercept = np.log(rate / (1.0 - rate))
        coefficient = np.zeros(stop - start, dtype=float)

        for _ in range(max_iter):
            probability = expit(
                np.clip(
                    intercept[None, :] + standardized * coefficient[None, :],
                    -35.0,
                    35.0,
                )
            )
            residual = np.where(valid, label_column - probability, 0.0)
            weight = np.where(valid, probability * (1.0 - probability), 0.0)
            gradient_intercept = residual.sum(axis=0)
            gradient_coefficient = (
                (residual * standardized).sum(axis=0)
                - l2_penalty * coefficient
            )
            h_ii = weight.sum(axis=0) + 1e-12
            h_ic = (weight * standardized).sum(axis=0)
            h_cc = (
                (weight * standardized * standardized).sum(axis=0)
                + l2_penalty
            )
            determinant = np.maximum(h_ii * h_cc - h_ic * h_ic, 1e-12)
            delta_intercept = (
                gradient_intercept * h_cc
                - gradient_coefficient * h_ic
            ) / determinant
            delta_coefficient = (
                gradient_coefficient * h_ii
                - gradient_intercept * h_ic
            ) / determinant
            delta_intercept = np.where(eligible, delta_intercept, 0.0)
            delta_coefficient = np.where(eligible, delta_coefficient, 0.0)
            intercept += np.clip(delta_intercept, -5.0, 5.0)
            coefficient += np.clip(delta_coefficient, -5.0, 5.0)
            if max(
                float(np.max(np.abs(delta_intercept))),
                float(np.max(np.abs(delta_coefficient))),
            ) < 1e-7:
                break

        slc = slice(start, stop)
        means[slc] = mean
        scales[slc] = safe_scale
        intercepts[slc] = np.where(eligible, intercept, np.nan)
        coefficients[slc] = np.where(eligible, coefficient, np.nan)
        eligible_all[slc] = eligible

    return means, scales, intercepts, coefficients, eligible_all


def _fast_binary_auc(labels, predictions):
    valid = np.isfinite(labels) & np.isfinite(predictions)
    labels = labels[valid]
    predictions = predictions[valid]
    if len(labels) < 10 or np.unique(labels).size < 2:
        return np.nan
    positive = labels == 1.0
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    ranks = rankdata(predictions, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _fast_screen_metrics(predictions, labels, features):
    rows = []
    for column, feature in enumerate(features):
        probability = predictions[:, column]
        valid = np.isfinite(probability) & np.isfinite(labels)
        p = probability[valid]
        y_feature = labels[valid]
        n = len(p)
        if n < 20 or np.unique(y_feature).size < 2:
            metrics = {
                "n": n,
                "auc": np.nan,
                "brier": np.nan,
                "accuracy": np.nan,
                "auc_first": np.nan,
                "auc_second": np.nan,
                "stability_gap": np.nan,
                "rank_score": np.nan,
            }
        else:
            midpoint = n // 2
            auc = _fast_binary_auc(y_feature, p)
            auc_first = _fast_binary_auc(y_feature[:midpoint], p[:midpoint])
            auc_second = _fast_binary_auc(y_feature[midpoint:], p[midpoint:])
            stability_gap = (
                abs(auc_first - auc_second)
                if np.isfinite(auc_first) and np.isfinite(auc_second)
                else 0.10
            )
            metrics = {
                "n": n,
                "auc": auc,
                "brier": float(np.mean((p - y_feature) ** 2)),
                "accuracy": float(np.mean((p >= 0.5) == y_feature)),
                "auc_first": auc_first,
                "auc_second": auc_second,
                "stability_gap": stability_gap,
                "rank_score": auc - 0.25 * stability_gap,
            }
        rows.append(
            {
                "feature": feature,
                "base": feature.split("__", 1)[0],
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False)


def screen_single_factors_fast_logistic(
    X,
    y,
    dev_end,
    min_train,
    horizon,
    eval_start=None,
    refit_every=3,
):
    """Causal expanding-window screen for a 10k-scale feature universe."""
    X = X.sort_index()
    y = y.reindex(X.index)
    eval_dates = X.index
    if eval_start is not None:
        eval_dates = eval_dates[eval_dates >= pd.Timestamp(eval_start)]
    if dev_end is not None:
        eval_dates = eval_dates[eval_dates <= pd.Timestamp(dev_end)]

    predictions = np.full((len(eval_dates), X.shape[1]), np.nan, dtype=float)
    feature_values = X.to_numpy(dtype=float, na_value=np.nan)
    index_periods = X.index.to_period("M")
    y_values = y.to_numpy(dtype=float, na_value=np.nan)
    state = None

    for position, test_date in enumerate(eval_dates):
        if state is None or position % refit_every == 0:
            cutoff_period = test_date.to_period("M") - horizon
            train_mask = (index_periods <= cutoff_period) & np.isfinite(y_values)
            state = _fit_fast_univariate_logistic(
                feature_values[train_mask],
                y_values[train_mask],
                min_train,
            )
        means, scales, intercepts, coefficients, eligible = state
        row_position = X.index.get_loc(test_date)
        current = feature_values[row_position]
        valid = eligible & np.isfinite(current)
        linear_score = intercepts[valid] + (
            (current[valid] - means[valid]) / scales[valid]
        ) * coefficients[valid]
        predictions[position, valid] = expit(np.clip(linear_score, -35.0, 35.0))

    labels = y.reindex(eval_dates).to_numpy(dtype=float, na_value=np.nan)
    return _fast_screen_metrics(predictions, labels, list(X.columns))


def screen_single_factors(
    X,
    y,
    dev_end,
    min_train,
    horizon,
    eval_start=None,
    refit_every=3,
    n_jobs=-1,
    model_type="logistic",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
):

    print(
        f"[ML] 단일 Factor ML 시작 "
        f"({X.shape[1]:,}개)"
    )

    if model_type == "fast_logistic":
        return screen_single_factors_fast_logistic(
            X=X,
            y=y,
            dev_end=dev_end,
            eval_start=eval_start,
            min_train=min_train,
            horizon=horizon,
            refit_every=refit_every,
        )

    results = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        verbose=10,
    )(
        delayed(_screen_one_factor)(
            feature,
            X,
            y,
            dev_end,
            eval_start,
            min_train,
            horizon,
            refit_every,
            model_type,
            svm_params,
            mlp_params,
            calibration_splits,
            random_state,
        )
        for feature in X.columns
    )

    scores = pd.DataFrame(
        results
    )

    scores = scores.sort_values(
        "rank_score",
        ascending=False,
    )

    return scores


def build_candidate_funnel(
    scores,
    feature_groups,
    *,
    max_per_base=3,
    max_per_group=15,
):
    """Compress transform families before correlation and combination search.

    Ranking is always inherited from the causal single-factor screen.  The
    first stage keeps several recipes for every raw sensor; the second stage
    prevents one economic basket from filling the entire search universe.
    """
    required = {"feature", "rank_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise KeyError(f"Candidate funnel columns missing: {sorted(missing)}")
    if max_per_base < 1 or max_per_group < 1:
        raise ValueError("Candidate funnel limits must be positive")

    ranked = scores.loc[scores["rank_score"].notna()].copy()
    ranked = ranked.sort_values("rank_score", ascending=False, kind="mergesort")
    ranked["base"] = ranked["feature"].str.split("__", n=1).str[0]
    ranked["group"] = ranked["base"].map(feature_groups)
    missing_group = ranked["group"].isna()
    ranked.loc[missing_group, "group"] = (
        "unknown:" + ranked.loc[missing_group, "base"].astype(str)
    )

    ranked["rank_within_base"] = ranked.groupby("base", sort=False).cumcount() + 1
    raw_stage = ranked.loc[ranked["rank_within_base"] <= max_per_base].copy()
    raw_stage["rank_after_raw_stage"] = np.arange(1, len(raw_stage) + 1)

    raw_stage["rank_within_group"] = (
        raw_stage.groupby("group", sort=False).cumcount() + 1
    )
    group_stage = raw_stage.loc[
        raw_stage["rank_within_group"] <= max_per_group
    ].copy()
    group_stage["rank_after_group_stage"] = np.arange(1, len(group_stage) + 1)
    return raw_stage, group_stage


def select_required_core_features(
    scores,
    required_families,
    *,
    min_oos_predictions,
    required_transform_tokens=None,
):
    """Select one causally screened transform from every required family."""
    required_transform_tokens = required_transform_tokens or {}
    ranked = scores.copy()
    ranked["base"] = ranked["feature"].str.split("__", n=1).str[0]
    rows = []
    selected = []
    for family, bases in required_families.items():
        candidates = ranked.loc[
            ranked["base"].isin(tuple(bases))
            & ranked["n"].ge(min_oos_predictions)
            & ranked["rank_score"].notna()
        ].copy()
        tokens = tuple(required_transform_tokens.get(family, ()))
        if tokens:
            suffix = candidates["feature"].map(
                lambda feature: feature.split("__", 1)[1]
                if "__" in feature
                else ""
            )
            candidates = candidates.loc[
                suffix.map(lambda value: any(token in value for token in tokens))
            ]
        if candidates.empty:
            raise RuntimeError(
                f"Required core family {family!r} has no eligible causal transform"
            )
        winner = candidates.sort_values(
            ["rank_score", "auc", "feature"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        selected.append(winner["feature"])
        rows.append(
            {
                "required_family": family,
                "feature": winner["feature"],
                "base": winner["base"],
                "screen_observations": int(winner["n"]),
                "screen_auc": float(winner["auc"]),
                "screen_auc_first": float(winner["auc_first"]),
                "screen_auc_second": float(winner["auc_second"]),
                "screen_rank_score": float(winner["rank_score"]),
                "transform_constraint": "|".join(tokens) if tokens else "any_causal",
            }
        )
    return selected, pd.DataFrame(rows)


def round_robin_group_candidates(
    ranked_features,
    feature_groups,
    *,
    max_features=8,
):
    """Build a small exhaustive pool without letting the first group crowd it."""
    if max_features < 1:
        raise ValueError("max_features must be positive")
    buckets = {}
    group_order = []
    for feature in ranked_features:
        base = feature.split("__", 1)[0]
        group = feature_groups.get(base, f"unknown:{base}")
        if group not in buckets:
            buckets[group] = []
            group_order.append(group)
        buckets[group].append(feature)

    selected = []
    depth = 0
    while len(selected) < max_features:
        added = False
        for group in group_order:
            bucket = buckets[group]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= max_features:
                    break
        if not added:
            break
        depth += 1
    return selected


# ============================================================
# CORRELATION PRUNE
# ============================================================

def prune_correlated_features(
    X,
    ranked_features,
    dev_end,
    threshold=0.90,
    max_features=40,
    min_obs=36,
):

    kept = []

    history = X.loc[
        :pd.Timestamp(dev_end)
    ]

    for feature in ranked_features:

        s = history[
            feature
        ]

        if s.count() < min_obs:
            continue

        reject = False

        for existing in kept:

            pair = history[
                [
                    feature,
                    existing,
                ]
            ].dropna()

            if len(pair) < min_obs:
                continue

            corr = pair.corr().iloc[
                0, 1
            ]

            if (
                np.isfinite(corr)
                and abs(corr)
                >= threshold
            ):

                reject = True
                break

        if not reject:
            kept.append(feature)

        if len(kept) >= max_features:
            break

    return kept


# ============================================================
# GREEDY COMBINATION SEARCH
# ============================================================

def greedy_forward_selection(
    X,
    y,
    candidates,
    validation_start,
    validation_end,
    min_train,
    horizon,
    max_features=6,
    min_improvement=0.002,
    max_features_per_base=None,
    feature_groups=None,
    max_features_per_group=None,
    model_type="logistic",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
    refit_every=1,
):

    selected = []

    best_score = 0.50

    search_history = []

    for step in range(
        max_features
    ):

        best_candidate = None
        best_candidate_score = -np.inf
        best_candidate_metrics = None

        for candidate in candidates:

            if candidate in selected:
                continue

            candidate_base = (
                candidate.split("__")[0]
            )

            current_bases = [
                f.split("__")[0]
                for f in selected
            ]

            if (
                max_features_per_base is not None
                and
                current_bases.count(
                    candidate_base
                )
                >= max_features_per_base
            ):
                continue

            # 동일 경제 그룹(Cycle/Liquidity/Credit/
            # Inflation/Risk)의 중복 선택도 제한한다.
            # 매핑이 없는 Factor는 서로 다른 unknown 그룹으로
            # 취급해 기존 호출의 동작을 보존한다.
            if feature_groups is not None and max_features_per_group is not None:

                candidate_group = (
                    feature_groups.get(
                        candidate_base,
                        f"unknown:{candidate_base}",
                    )
                )

                current_groups = [
                    feature_groups.get(
                        feature.split("__")[0],
                        (
                            "unknown:"
                            f"{feature.split('__')[0]}"
                        ),
                    )
                    for feature in selected
                ]

                if (
                    current_groups.count(
                        candidate_group
                    )
                    >= max_features_per_group
                ):
                    continue

            trial_features = (
                selected
                + [candidate]
            )

            prediction = (
                walk_forward_predict(
                    X[trial_features],
                    y,
                    eval_start=validation_start,
                    eval_end=validation_end,
                    min_train=min_train,
                    purge=horizon,
                    refit_every=refit_every,
                    model_type=model_type,
                    svm_params=svm_params,
                    mlp_params=mlp_params,
                    calibration_splits=calibration_splits,
                    random_state=random_state,
                )
            )

            metrics = (
                evaluate_probabilities(
                    prediction,
                    y,
                )
            )

            auc = metrics["auc"]

            if not np.isfinite(auc):
                continue

            search_history.append({
                "step": step + 1,
                "candidate": candidate,
                "features": "|".join(
                    trial_features
                ),
                **metrics,
            })

            if (
                auc
                > best_candidate_score
            ):

                best_candidate = candidate
                best_candidate_score = auc
                best_candidate_metrics = metrics

        if best_candidate is None:
            break

        improvement = (
            best_candidate_score
            - best_score
        )

        # 첫 Factor는 일단 선택
        if (
            selected
            and improvement
            < min_improvement
        ):

            print(
                "🛑 추가 Factor가 "
                "Validation 성능을 "
                "충분히 개선하지 못함."
            )

            break

        selected.append(
            best_candidate
        )

        best_score = (
            best_candidate_score
        )

        print(
            f"[OK] Step {step + 1}: "
            f"{best_candidate}"
        )

        print(
            f"   Validation AUC = "
            f"{best_score:.4f}"
        )

    return (
        selected,
        pd.DataFrame(
            search_history
        ),
    )


def exhaustive_combination_selection(
    X,
    y,
    candidates,
    validation_start,
    validation_end,
    min_train,
    horizon,
    *,
    min_features=1,
    max_features=3,
    max_features_per_base=None,
    feature_groups=None,
    max_features_per_group=None,
    min_distinct_groups=1,
    model_type="svm_rank",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
    refit_every=3,
):
    """Evaluate every valid subset of a pre-reduced 8--12 feature pool."""
    if len(candidates) > 12:
        raise ValueError("Exhaustive search is restricted to at most 12 candidates")
    history = []
    best_features = []
    best_key = (-np.inf, -np.inf, 0, "")
    upper_size = min(max_features, len(candidates))
    if min_features < 1 or min_features > max_features:
        raise ValueError("min_features must be between 1 and max_features")
    for size in range(min_features, upper_size + 1):
        for trial in combinations(candidates, size):
            bases = [feature.split("__", 1)[0] for feature in trial]
            if max_features_per_base is not None and any(
                bases.count(base) > max_features_per_base for base in set(bases)
            ):
                continue
            if feature_groups is not None and max_features_per_group is not None:
                groups = [
                    feature_groups.get(base, f"unknown:{base}") for base in bases
                ]
                if any(
                    groups.count(group) > max_features_per_group
                    for group in set(groups)
                ):
                    continue
            elif feature_groups is not None:
                groups = [
                    feature_groups.get(base, f"unknown:{base}") for base in bases
                ]
            else:
                groups = []
            if min_distinct_groups > 1:
                if feature_groups is None:
                    raise ValueError(
                        "feature_groups is required when min_distinct_groups > 1"
                    )
                if len(set(groups)) < min_distinct_groups:
                    continue
            prediction = walk_forward_predict(
                X[list(trial)],
                y,
                eval_start=validation_start,
                eval_end=validation_end,
                min_train=min_train,
                purge=horizon,
                refit_every=refit_every,
                model_type=model_type,
                svm_params=svm_params,
                mlp_params=mlp_params,
                calibration_splits=calibration_splits,
                random_state=random_state,
            )
            metrics = evaluate_probabilities(prediction, y)
            history.append(
                {
                    "search": "bounded_exhaustive",
                    "combination_size": size,
                    "features": "|".join(trial),
                    **metrics,
                }
            )
            rank_score = metrics.get("rank_score", np.nan)
            auc = metrics.get("auc", np.nan)
            if not np.isfinite(rank_score) or not np.isfinite(auc):
                continue
            # Prefer validation stability score, then AUC, then smaller models.
            key = (rank_score, auc, -size, "|".join(trial))
            if key > best_key:
                best_key = key
                best_features = list(trial)
    return best_features, pd.DataFrame(history)


# ============================================================
# DATE SPLIT
# ============================================================

def chronological_split(
    y,
    development_fraction=0.60,
    validation_fraction=0.20,
    test_start=None,
    validation_months=None,
):

    idx = y.dropna().index

    n = len(idx)

    if n < 100:
        raise ValueError(
            "Target 데이터가 너무 짧아."
        )

    if test_start is not None:
        configured_test_start = pd.Timestamp(test_start).to_period("M").to_timestamp("M")
        matching = np.flatnonzero(idx == configured_test_start)
        if len(matching) != 1:
            raise ValueError(
                "Configured test_start must match exactly one target month: "
                f"{configured_test_start.date()}"
            )
        val_pos = int(matching[0])
        if validation_months is None:
            validation_months = max(1, int(round(val_pos * 0.25)))
        dev_pos = val_pos - int(validation_months)
        if dev_pos < 1:
            raise ValueError("Not enough history before the configured validation window")
    else:
        dev_pos = int(n * development_fraction)
        val_pos = int(n * (development_fraction + validation_fraction))

    dev_end = idx[
        dev_pos - 1
    ]

    validation_start = idx[
        dev_pos
    ]

    validation_end = idx[
        val_pos - 1
    ]

    test_start = idx[
        val_pos
    ]

    test_end = idx[-1]

    return {
        "dev_end": dev_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
    }


# ============================================================
# LATEST EWS
# ============================================================
def fit_latest_ews(
    X,
    y,
    features,
    horizon,
    asof_date=None,
    min_train=60,
    model_type="logistic",
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
    market_name="KOSPI200",
    max_train=None,
):

    # ========================================================
    # 1. 선택된 Factor
    # ========================================================

    X_selected = (
        X[features]
        .sort_index()
        .copy()
    )

    # ========================================================
    # ★ 핵심 수정
    #
    # X = FRED 전체기간 → 예: 955개월
    # y = KOSPI 존재기간 → 예: 199개월
    #
    # y를 X 날짜축에 맞춰놓는다.
    # KOSPI가 없던 옛날 날짜는 NaN이 되고
    # 나중에 dropna()로 자동 제거된다.
    # ========================================================

    y_aligned = (
        y.reindex(
            X_selected.index
        )
        .copy()
    )


    # ========================================================
    # 2. 현재 시점에서 모든 Factor가 존재하는 행
    # ========================================================

    complete = (
        X_selected
        .dropna()
    )

    if asof_date is not None:

        asof_date = pd.Timestamp(
            asof_date
        )

        complete = (
            complete.loc[
                :asof_date
            ]
        )


    if complete.empty:

        raise ValueError(
            "현재 시점에서 모든 선택 Factor가 "
            "동시에 존재하는 날짜가 없어."
        )


    # ========================================================
    # 3. 최신 예측 날짜
    # ========================================================

    latest_date = (
        complete.index.max()
    )

    latest_x = (
        complete.loc[
            [latest_date]
        ]
    )


    # ========================================================
    # 4. PURGE
    #
    # 예:
    # horizon = 3
    #
    # 2025-12월에 예측한다면
    # 2025-09월 Target까지만 학습 가능
    #
    # 최근 3개월 target은 미래 가격이 필요하니까
    # 학습에 넣으면 안 된다.
    # ========================================================

    cutoff_date = (
        (
            latest_date.to_period("M")
            - horizon
        )
        .to_timestamp("M")
    )


    # ========================================================
    # 5. X와 y를 같은 날짜범위로 자른 뒤 concat
    #
    # 이전 코드처럼
    #
    # X mask 길이 = 955
    # y 길이      = 199
    #
    # 상태에서 boolean mask를 공유하지 않는다.
    # ========================================================

    train_X = (
        X_selected.loc[
            :cutoff_date
        ]
    )

    train_y = (
        y_aligned.loc[
            :cutoff_date
        ]
    )


    train = pd.concat(
        [
            train_X,
            train_y.rename("y"),
        ],
        axis=1,
    ).dropna()

    if max_train is not None:
        if max_train < min_train:
            raise ValueError("max_train must be at least min_train")
        train = train.tail(max_train)


    # ========================================================
    # 6. 검사
    # ========================================================

    if len(train) < min_train:

        raise ValueError(
            f"최종 모델 학습 데이터 부족: "
            f"{len(train)}개월 "
            f"(필요 최소 {min_train}개월)"
        )


    if train["y"].nunique() < 2:

        raise ValueError(
            "Target에 0/1 두 클래스가 모두 없어."
        )


    # ========================================================
    # 7. MODEL
    # ========================================================

    model = make_model(
        model_type=model_type,
        horizon=horizon,
        y_train=train["y"],
        svm_params=svm_params,
        mlp_params=mlp_params,
        calibration_splits=calibration_splits,
        random_state=random_state,
    )


    # ========================================================
    # 8. FIT
    # ========================================================

    fit_classification_model(
        model,
        train[features],
        train["y"],
    )


    # ========================================================
    # 9. 현재 Risk-On 확률
    # ========================================================

    probability = (
        model.predict_proba(
            latest_x[features]
        )[0, 1]
    )

    ews = (
        probability
        * 100
    )


    # ========================================================
    # 10. DEBUG 정보
    # ========================================================

    print()
    print("[LATEST] Latest EWS Training Info")
    print(
        f"FRED 전체 기간      : "
        f"{X_selected.index.min().date()} "
        f"~ {X_selected.index.max().date()}"
    )

    y_valid = (
        y_aligned.dropna()
    )

    print(
        f"{market_name} Target 기간   : "
        f"{y_valid.index.min().date()} "
        f"~ {y_valid.index.max().date()}"
    )

    print(
        f"예측 기준일          : "
        f"{latest_date.date()}"
    )

    print(
        f"학습 Target 마감일   : "
        f"{cutoff_date.date()}"
    )

    print(
        f"실제 학습 표본       : "
        f"{len(train)}개월"
    )

    print(
        f"최종 EWS             : "
        f"{ews:.2f}"
    )


    return {
        "date": latest_date,

        "probability":
            probability,

        "ews":
            probability * 100,

        "model":
            model,

        "model_type":
            model_type,

        "train_n":
            len(train),

        "train_start":
            train.index.min(),

        "train_end":
            train.index.max(),

        "features":
            features,

        "feature_values":
            latest_x.T.rename(
                columns={
                    latest_date:
                        "value"
                }
            ),
    }


def ews_state(
    score,
    risk_off=35,
    risk_on=65,
):

    if score < risk_off:
        return "Risk-Off"

    if score >= risk_on:
        return "Risk-On"

    return "Neutral"
