"""Purged research folds and allocation-policy evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.analytics import performance_stats
from src.backtest import run_backtest
from src.position_sizing import target_weight_from_ews
from src.modeling import (
    build_candidate_funnel,
    earliest_walk_forward_prediction_date,
    exhaustive_combination_selection,
    greedy_forward_selection,
    make_model,
    prune_correlated_features,
    round_robin_group_candidates,
    screen_single_factors,
    select_required_core_features,
    walk_forward_predict,
)


@dataclass(frozen=True)
class OuterFold:
    fold: int
    development_end: pd.Timestamp
    inner_validation_start: pd.Timestamp
    inner_validation_end: pd.Timestamp
    outer_start: pd.Timestamp
    outer_end: pd.Timestamp


def make_purged_outer_folds(
    target_index,
    *,
    research_end,
    min_train_months,
    inner_validation_months=24,
    outer_validation_months=24,
    purge_months=3,
    screening_oos_months=36,
):
    """Construct expanding, non-overlapping outer folds before a cutoff.

    Each fold reserves an inner validation window and a purge interval before
    the outer window.  All dates are taken from the observed target index so
    no calendar assumptions leak into the split.
    """
    index = pd.DatetimeIndex(target_index).sort_values().unique()
    index = index[index <= pd.Timestamp(research_end)]
    # A screening prediction at position t can only use labels through
    # t - purge.  Therefore the development window needs M training rows,
    # the purge gap, and S genuinely out-of-sample predictions.  The old
    # formula omitted ``purge_months - 1`` here, so the first fold could
    # produce only 33 predictions while requiring 36.
    minimum_development_observations = (
        min_train_months + purge_months + screening_oos_months - 1
    )
    first_outer = (
        minimum_development_observations
        + inner_validation_months
        + 2 * purge_months
    )
    if len(index) <= first_outer:
        raise ValueError("Not enough pre-holdout observations for nested validation")

    folds = []
    fold_number = 1
    for outer_position in range(first_outer, len(index), outer_validation_months):
        outer_end_position = min(
            outer_position + outer_validation_months - 1,
            len(index) - 1,
        )
        inner_end_position = outer_position - purge_months - 1
        inner_start_position = inner_end_position - inner_validation_months + 1
        development_end_position = inner_start_position - purge_months - 1
        if development_end_position < minimum_development_observations - 1:
            continue
        folds.append(
            OuterFold(
                fold=fold_number,
                development_end=index[development_end_position],
                inner_validation_start=index[inner_start_position],
                inner_validation_end=index[inner_end_position],
                outer_start=index[outer_position],
                outer_end=index[outer_end_position],
            )
        )
        fold_number += 1
    if not folds:
        raise ValueError("No valid outer folds were generated")
    return folds


def screening_evaluation_start(
    target_index,
    *,
    development_end,
    min_train_months,
    purge_months,
    min_oos_predictions,
    evaluation_window_months=60,
):
    """Return a screen start where the very first refit is trainable.

    Starting a fixed trailing window before enough purged labels exist makes
    the refit schedule waste its first iterations and can silently reduce the
    OOS count.  This helper clamps that window to the first date with the full
    training sample and verifies the promised OOS count before fitting.
    """
    index = pd.DatetimeIndex(target_index).sort_values().unique()
    index = index[index <= pd.Timestamp(development_end)]
    if min_train_months < 1 or purge_months < 0 or min_oos_predictions < 1:
        raise ValueError("Screening window parameters are invalid")
    first_trainable_position = min_train_months + purge_months - 1
    trailing_window_position = max(
        0,
        len(index) - max(evaluation_window_months, min_oos_predictions),
    )
    start_position = max(first_trainable_position, trailing_window_position)
    available_predictions = len(index) - start_position
    if start_position >= len(index) or available_predictions < min_oos_predictions:
        raise ValueError(
            "Development window cannot supply the required purged screening "
            f"predictions: available={max(available_predictions, 0)}, "
            f"required={min_oos_predictions}"
        )
    return index[start_position]


def fold_candidate_availability_audit(
    *,
    X,
    y,
    folds,
    min_train_months,
    horizon,
    min_oos_predictions,
    required_families=None,
    required_transform_tokens=None,
    minimum_available_features=1,
):
    """Declare folds that can be fit before looking at prediction quality.

    Eligibility uses only feature/label presence, two-class trainability and
    the predeclared sample sizes.  AUC, IC and returns are never consulted.
    """
    if minimum_available_features < 1:
        raise ValueError("minimum_available_features must be positive")
    features = X.sort_index()
    labels = y.reindex(features.index)
    periods = features.index.to_period("M")
    values = features.to_numpy(dtype=float, na_value=np.nan)
    label_values = labels.to_numpy(dtype=float, na_value=np.nan)
    feature_present = np.isfinite(values)
    label_present = np.isfinite(label_values)
    train_present = feature_present & label_present[:, None]
    cumulative_n = np.cumsum(train_present, axis=0)
    cumulative_zero = np.cumsum(
        train_present & (label_values[:, None] == 0), axis=0
    )
    cumulative_one = np.cumsum(
        train_present & (label_values[:, None] == 1), axis=0
    )

    required_transform_tokens = required_transform_tokens or {}
    if required_families:
        scopes = []
        for family, bases in required_families.items():
            tokens = tuple(required_transform_tokens.get(family, ()))
            columns = []
            for position, feature in enumerate(features.columns):
                base, _, suffix = feature.partition("__")
                if base not in tuple(bases):
                    continue
                if tokens and not any(token in suffix for token in tokens):
                    continue
                columns.append(position)
            scopes.append((family, columns, 1))
    else:
        scopes = [
            ("all_candidates", list(range(features.shape[1])), minimum_available_features)
        ]

    rows = []
    for fold in folds:
        screen_start = screening_evaluation_start(
            labels.dropna().index,
            development_end=fold.development_end,
            min_train_months=min_train_months,
            purge_months=horizon,
            min_oos_predictions=min_oos_predictions,
        )
        eval_positions = np.flatnonzero(
            (features.index >= screen_start)
            & (features.index <= fold.development_end)
        )
        cutoff_positions = np.searchsorted(
            periods.asi8,
            (features.index[eval_positions].to_period("M") - horizon).asi8,
            side="right",
        ) - 1
        valid_cutoff = cutoff_positions >= 0
        possible = np.zeros(features.shape[1], dtype=int)
        if valid_cutoff.any():
            eval_positions = eval_positions[valid_cutoff]
            cutoff_positions = cutoff_positions[valid_cutoff]
            fit_ok = (
                (cumulative_n[cutoff_positions] >= min_train_months)
                & (cumulative_zero[cutoff_positions] > 0)
                & (cumulative_one[cutoff_positions] > 0)
            )
            current_ok = (
                feature_present[eval_positions]
                & label_present[eval_positions, None]
            )
            possible = (fit_ok & current_ok).sum(axis=0)

        fold_scope_passes = []
        fold_rows = []
        for scope, columns, required_count in scopes:
            counts = possible[columns] if columns else np.array([], dtype=int)
            eligible_count = int((counts >= min_oos_predictions).sum())
            passed = eligible_count >= required_count
            fold_scope_passes.append(passed)
            fold_rows.append(
                {
                    "fold": fold.fold,
                    "development_end": fold.development_end,
                    "scope": scope,
                    "candidate_features": len(columns),
                    "required_available_features": required_count,
                    "eligible_features": eligible_count,
                    "max_possible_predictions": int(counts.max()) if len(counts) else 0,
                    "availability_passed": passed,
                }
            )
        fold_passed = bool(fold_scope_passes and all(fold_scope_passes))
        for row in fold_rows:
            row["fold_availability_passed"] = fold_passed
            row["exclusion_reason"] = (
                "eligible_for_model_validation"
                if fold_passed
                else "pre_model_inception_or_insufficient_feature_history"
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_fold_availability_audit(
    *, X, y, folds, min_train_months, horizon
):
    """Audit outer-fold availability for an already frozen feature set."""
    inception = earliest_walk_forward_prediction_date(
        X,
        y,
        min_train=min_train_months,
        purge=horizon,
    )
    rows = []
    for fold in folds:
        evaluation_start = (
            max(fold.outer_start, inception) if inception is not None else None
        )
        expected = int(
            y.loc[evaluation_start : fold.outer_end].notna().sum()
            if evaluation_start is not None and evaluation_start <= fold.outer_end
            else 0
        )
        passed = expected > 0
        rows.append(
            {
                "fold": fold.fold,
                "outer_start": fold.outer_start,
                "outer_end": fold.outer_end,
                "model_eligibility_start": inception,
                "evaluation_start": evaluation_start,
                "expected_observations": expected,
                "fold_availability_passed": passed,
                "exclusion_reason": (
                    "eligible_for_model_validation"
                    if passed
                    else "pre_model_inception_or_insufficient_feature_history"
                ),
            }
        )
    return pd.DataFrame(rows)


def nested_outer_predict(
    *,
    X,
    y,
    folds,
    feature_groups,
    screening_model_type,
    selection_model_type,
    final_model_type,
    min_train_months,
    final_min_train_months,
    horizon,
    single_factor_refit_every,
    min_oos_predictions,
    top_feature_pool,
    raw_top_features_per_base,
    group_candidates_per_group,
    correlation_threshold,
    combination_candidate_pool,
    exhaustive_candidate_pool,
    min_model_features,
    max_model_features,
    min_validation_improvement,
    max_features_per_base,
    max_features_per_group,
    min_distinct_groups,
    svm_params,
    mlp_params,
    calibration_splits,
    random_state=42,
    combination_refit_every=3,
    n_jobs=-1,
    final_refit_every=1,
    required_core_families=None,
    required_core_transform_tokens=None,
    allow_unavailable_folds=False,
    final_max_train_months=None,
):
    """Repeat the full screen/select/fit process inside every outer fold."""
    predictions = []
    selections = []
    screening_rows = []

    for fold in folds:
        print(
            f"\nOuter fold {fold.fold}: {fold.outer_start.date()} "
            f"~ {fold.outer_end.date()}"
        )
        screen_start = screening_evaluation_start(
            y.dropna().index,
            development_end=fold.development_end,
            min_train_months=min_train_months,
            purge_months=horizon,
            min_oos_predictions=min_oos_predictions,
        )
        scores = screen_single_factors(
            X=X,
            y=y,
            dev_end=fold.development_end,
            eval_start=screen_start,
            min_train=min_train_months,
            horizon=horizon,
            refit_every=single_factor_refit_every,
            n_jobs=n_jobs,
            model_type=screening_model_type,
            svm_params=svm_params,
            mlp_params=mlp_params,
            calibration_splits=calibration_splits,
            random_state=random_state,
        )
        scores["fold"] = fold.fold
        scores["development_end"] = fold.development_end
        screening_rows.append(scores.head(top_feature_pool))

        usable = scores[
            (scores["n"] >= min_oos_predictions) & scores["rank_score"].notna()
        ]
        if required_core_families:
            selected, core_audit = select_required_core_features(
                usable,
                required_core_families,
                min_oos_predictions=min_oos_predictions,
                required_transform_tokens=required_core_transform_tokens,
            )
            core_audit = core_audit.assign(
                fold=fold.fold,
                development_end=fold.development_end,
                selection_scope="development_screen_only",
            )
            screening_rows.append(core_audit)
            history = pd.DataFrame()
        else:
            selected = None
            _, balanced = build_candidate_funnel(
                usable,
                feature_groups,
                max_per_base=raw_top_features_per_base,
                max_per_group=group_candidates_per_group,
            )
            ranked = balanced["feature"].tolist()
            candidates = prune_correlated_features(
                X=X,
                ranked_features=ranked,
                dev_end=fold.development_end,
                threshold=correlation_threshold,
                max_features=combination_candidate_pool,
                min_obs=min_oos_predictions,
            )
            if not candidates:
                if allow_unavailable_folds:
                    screening_rows.append(
                        pd.DataFrame(
                            [
                                {
                                    "fold": fold.fold,
                                    "development_end": fold.development_end,
                                    "selection_status": "pre_model_inception_no_candidates",
                                }
                            ]
                        )
                    )
                    continue
                raise RuntimeError(f"Outer fold {fold.fold} has no eligible candidates")
            exhaustive_candidates = round_robin_group_candidates(
                candidates,
                feature_groups,
                max_features=exhaustive_candidate_pool,
            )
            if len(exhaustive_candidates) < min_model_features:
                if allow_unavailable_folds:
                    screening_rows.append(
                        pd.DataFrame(
                            [
                                {
                                    "fold": fold.fold,
                                    "development_end": fold.development_end,
                                    "selection_status": (
                                        "pre_model_inception_insufficient_candidates"
                                    ),
                                    "eligible_candidates": len(exhaustive_candidates),
                                }
                            ]
                        )
                    )
                    continue
                raise RuntimeError(
                    f"Outer fold {fold.fold} has only {len(exhaustive_candidates)} "
                    "eligible candidates after screening"
                )
            selected, history = exhaustive_combination_selection(
                X=X,
                y=y,
                candidates=exhaustive_candidates,
                validation_start=fold.inner_validation_start,
                validation_end=fold.inner_validation_end,
                min_train=final_min_train_months,
                horizon=horizon,
                min_features=min_model_features,
                max_features=max_model_features,
                max_features_per_base=max_features_per_base,
                feature_groups=feature_groups,
                max_features_per_group=max_features_per_group,
                min_distinct_groups=min_distinct_groups,
                model_type=selection_model_type,
                svm_params=svm_params,
                mlp_params=mlp_params,
                calibration_splits=calibration_splits,
                random_state=random_state,
                refit_every=combination_refit_every,
            )
        if not selected:
            if allow_unavailable_folds:
                if not history.empty:
                    history = history.assign(fold=fold.fold)
                    screening_rows.append(history)
                screening_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "fold": fold.fold,
                                "development_end": fold.development_end,
                                "selection_status": (
                                    "pre_model_inception_no_evaluable_combination"
                                ),
                            }
                        ]
                    )
                )
                continue
            raise RuntimeError(f"Outer fold {fold.fold} has no valid feature set")

        model_eligibility_start = earliest_walk_forward_prediction_date(
            X[selected],
            y,
            min_train=final_min_train_months,
            purge=horizon,
        )
        if model_eligibility_start is None:
            raise RuntimeError(
                f"Outer fold {fold.fold} selected features never supply the "
                "declared purged training sample"
            )

        prediction = walk_forward_predict(
            X[selected],
            y,
            eval_start=max(fold.outer_start, model_eligibility_start),
            eval_end=fold.outer_end,
            min_train=final_min_train_months,
            purge=horizon,
            refit_every=final_refit_every,
            model_type=final_model_type,
            svm_params=svm_params,
            mlp_params=mlp_params,
            calibration_splits=calibration_splits,
            random_state=random_state,
            max_train=final_max_train_months,
        )
        predictions.append(prediction)
        for rank, feature in enumerate(selected, start=1):
            selections.append(
                {
                    "fold": fold.fold,
                    "development_end": fold.development_end,
                    "inner_validation_start": fold.inner_validation_start,
                    "inner_validation_end": fold.inner_validation_end,
                    "outer_start": fold.outer_start,
                    "outer_end": fold.outer_end,
                    "selection_rank": rank,
                    "feature": feature,
                    "base": feature.split("__", 1)[0],
                    "group": feature_groups.get(feature.split("__", 1)[0], "unknown"),
                    "model_eligibility_start": model_eligibility_start,
                }
            )
        if not history.empty:
            history = history.assign(fold=fold.fold)

    if not predictions:
        raise RuntimeError("No outer fold produced an evaluable prediction")
    oos_prediction = pd.concat(predictions).sort_index()
    oos_prediction = oos_prediction[~oos_prediction.index.duplicated(keep="last")]
    oos_prediction.name = f"{final_model_type}_nested_outer_prediction"
    return (
        oos_prediction,
        pd.DataFrame(selections),
        pd.concat(screening_rows, ignore_index=True),
    )


def fixed_outer_predict(
    *,
    X,
    y,
    folds,
    features,
    feature_groups,
    final_model_type,
    final_min_train_months,
    horizon,
    refit_every=1,
    svm_params=None,
    mlp_params=None,
    calibration_splits=3,
    random_state=42,
    selection_note="predeclared_structural_feature_set",
    final_max_train_months=None,
):
    """Score a feature set fixed independently of every outer-fold outcome."""
    features = list(features)
    missing = [feature for feature in features if feature not in X]
    if missing:
        raise KeyError(f"Fixed outer features are missing: {missing}")
    model_eligibility_start = earliest_walk_forward_prediction_date(
        X[features],
        y,
        min_train=final_min_train_months,
        purge=horizon,
    )
    if model_eligibility_start is None:
        raise RuntimeError("Fixed feature set never supplies the purged training sample")

    predictions = []
    selections = []
    for fold in folds:
        prediction = walk_forward_predict(
            X[features],
            y,
            eval_start=max(fold.outer_start, model_eligibility_start),
            eval_end=fold.outer_end,
            min_train=final_min_train_months,
            purge=horizon,
            refit_every=refit_every,
            model_type=final_model_type,
            svm_params=svm_params,
            mlp_params=mlp_params,
            calibration_splits=calibration_splits,
            random_state=random_state,
            max_train=final_max_train_months,
        )
        predictions.append(prediction)
        for rank, feature in enumerate(features, start=1):
            selections.append(
                {
                    "fold": fold.fold,
                    "development_end": fold.development_end,
                    "inner_validation_start": fold.inner_validation_start,
                    "inner_validation_end": fold.inner_validation_end,
                    "outer_start": fold.outer_start,
                    "outer_end": fold.outer_end,
                    "selection_rank": rank,
                    "feature": feature,
                    "base": feature.split("__", 1)[0],
                    "group": feature_groups.get(
                        feature.split("__", 1)[0], "unknown"
                    ),
                    "model_eligibility_start": model_eligibility_start,
                    "selection_note": selection_note,
                }
            )
    oos_prediction = pd.concat(predictions).sort_index()
    oos_prediction = oos_prediction[~oos_prediction.index.duplicated(keep="last")]
    oos_prediction.name = f"{final_model_type}_fixed_outer_prediction"
    return oos_prediction, pd.DataFrame(selections), pd.DataFrame()


def calibration_diagnostics(prediction, y, bins=10):
    """Return calibration slope/intercept and a quantile reliability table."""
    data = pd.concat(
        [prediction.rename("probability"), y.rename("y")], axis=1
    ).dropna()
    if data.empty:
        raise ValueError("No observations for calibration diagnostics")

    clipped = data["probability"].clip(1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).to_numpy().reshape(-1, 1)
    if data["y"].nunique() < 2:
        calibration_intercept = np.nan
        calibration_slope = np.nan
    else:
        calibration = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
        ).fit(logit, data["y"].astype(int).to_numpy())
        calibration_intercept = float(calibration.intercept_[0])
        calibration_slope = float(calibration.coef_[0, 0])
    summary = {
        "observations": len(data),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "mean_probability": float(data["probability"].mean()),
        "observed_rate": float(data["y"].mean()),
    }

    effective_bins = min(bins, data["probability"].nunique())
    if effective_bins < 2:
        reliability = pd.DataFrame(
            {
                "bin": [0],
                "n": [len(data)],
                "mean_probability": [data["probability"].mean()],
                "observed_rate": [data["y"].mean()],
            }
        )
    else:
        bucket = pd.qcut(
            data["probability"],
            q=effective_bins,
            labels=False,
            duplicates="drop",
        )
        reliability = (
            data.assign(bin=bucket)
            .groupby("bin", observed=True)
            .agg(
                n=("y", "size"),
                mean_probability=("probability", "mean"),
                observed_rate=("y", "mean"),
            )
            .reset_index()
        )
    return summary, reliability


def logistic_fold_coefficient_audit(
    *,
    X,
    y,
    folds,
    selections,
    horizon,
    min_train_months,
    selection_scope="nested_outer_fold",
):
    """Fit the first model in each outer fold and expose coefficient signs.

    Coefficients are diagnostics for the fitted standardized Logistic model;
    they are not causal effects or a human economic-direction approval.
    """
    rows = []
    if selections.empty:
        return pd.DataFrame()
    for fold in folds:
        fold_selection = selections.loc[selections["fold"].eq(fold.fold)].copy()
        if fold_selection.empty:
            continue
        fold_selection = fold_selection.sort_values("selection_rank")
        features = fold_selection["feature"].tolist()
        cutoff_period = fold.outer_start.to_period("M") - horizon
        train_mask = X.index.to_period("M") <= cutoff_period
        train_index = X.index[train_mask]
        train = pd.concat(
            [
                X.loc[train_index, features],
                y.reindex(train_index).rename("y"),
            ],
            axis=1,
        ).dropna()
        if len(train) < min_train_months or train["y"].nunique() < 2:
            for item in fold_selection.itertuples(index=False):
                rows.append(
                    {
                        "fold": fold.fold,
                        "outer_start": fold.outer_start,
                        "outer_end": fold.outer_end,
                        "train_end": train.index.max() if not train.empty else None,
                        "train_observations": len(train),
                        "feature": item.feature,
                        "base": item.base,
                        "group": item.group,
                        "standardized_coefficient": np.nan,
                        "raw_unit_coefficient": np.nan,
                        "coefficient_sign": "unavailable",
                        "fit_status": "insufficient_training_data",
                        "selection_scope": selection_scope,
                        "interpretation": "diagnostic association; not causal approval",
                    }
                )
            continue

        model = make_model("logistic")
        model.fit(train[features], train["y"])
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["model"]
        standardized = classifier.coef_[0]
        raw_unit = standardized / scaler.scale_
        for item, scaled_coef, raw_coef in zip(
            fold_selection.itertuples(index=False), standardized, raw_unit
        ):
            rows.append(
                {
                    "fold": fold.fold,
                    "outer_start": fold.outer_start,
                    "outer_end": fold.outer_end,
                    "train_end": train.index.max(),
                    "train_observations": len(train),
                    "feature": item.feature,
                    "base": item.base,
                    "group": item.group,
                    "standardized_coefficient": float(scaled_coef),
                    "raw_unit_coefficient": float(raw_coef),
                    "coefficient_sign": (
                        "positive" if scaled_coef > 0 else "negative" if scaled_coef < 0 else "zero"
                    ),
                    "fit_status": "fitted",
                    "selection_scope": selection_scope,
                    "interpretation": "diagnostic association; not causal approval",
                }
            )
    return pd.DataFrame(rows)


def coefficient_sign_stability(coefficient_audit):
    """Summarize sign consistency only for features recurring across folds."""
    if coefficient_audit.empty:
        return pd.DataFrame()
    fitted = coefficient_audit.loc[
        coefficient_audit["fit_status"].eq("fitted")
        & coefficient_audit["coefficient_sign"].isin(["positive", "negative"])
    ]
    rows = []
    for (feature, base, group), data in fitted.groupby(
        ["feature", "base", "group"], sort=True
    ):
        positive = int(data["coefficient_sign"].eq("positive").sum())
        negative = int(data["coefficient_sign"].eq("negative").sum())
        observations = positive + negative
        dominant = max(positive, negative)
        rows.append(
            {
                "feature": feature,
                "base": base,
                "group": group,
                "fold_occurrences": observations,
                "positive_folds": positive,
                "negative_folds": negative,
                "dominant_sign": "positive" if positive >= negative else "negative",
                "sign_consistency_ratio": dominant / observations,
                "enough_recurrence_to_assess": observations >= 2,
                "interpretation": "diagnostic association; not causal approval",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fold_occurrences", "sign_consistency_ratio"], ascending=[False, True]
    )


def coefficient_family_sign_stability(coefficient_audit, required_families):
    """Summarize coefficient signs across changing transforms of one family.

    Every transform used here preserves the base indicator's increasing-value
    orientation.  This is a model-association diagnostic, not causal or human
    economic approval and therefore does not silently become a signal gate.
    """
    if coefficient_audit.empty:
        return pd.DataFrame()
    family_by_base = {
        base: family
        for family, bases in required_families.items()
        for base in bases
    }
    fitted = coefficient_audit.loc[
        coefficient_audit["fit_status"].eq("fitted")
        & coefficient_audit["coefficient_sign"].isin(["positive", "negative"])
    ].copy()
    fitted["required_family"] = fitted["base"].map(family_by_base)
    fitted = fitted.dropna(subset=["required_family"])
    rows = []
    for family, data in fitted.groupby("required_family", sort=True):
        positive = int(data["coefficient_sign"].eq("positive").sum())
        negative = int(data["coefficient_sign"].eq("negative").sum())
        observations = positive + negative
        dominant = max(positive, negative)
        rows.append(
            {
                "required_family": family,
                "fold_occurrences": observations,
                "distinct_transforms": int(data["feature"].nunique()),
                "positive_folds": positive,
                "negative_folds": negative,
                "dominant_sign": (
                    "positive" if positive >= negative else "negative"
                ),
                "sign_consistency_ratio": dominant / observations,
                "enough_recurrence_to_assess": observations >= 2,
                "interpretation": (
                    "family-level fitted association across orientation-preserving "
                    "transforms; diagnostic, not causal approval"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("required_family").reset_index(drop=True)


def evaluate_signal_gate(
    fold_signal,
    *,
    aggregate_auc,
    aggregate_rank_ic,
    minimum_fold_observations=20,
    minimum_joint_pass_ratio=2 / 3,
):
    """Return transparent fold checks and the predeclared signal decision."""
    details = fold_signal.copy()
    details["minimum_observations"] = minimum_fold_observations
    details["gate_eligible_fold"] = (
        details["expected_observations"] >= minimum_fold_observations
    )
    details["coverage_passed"] = (
        ~details["gate_eligible_fold"]
        | (details["observations"] >= minimum_fold_observations)
    )
    details["auc_threshold_passed"] = (
        np.isfinite(details["auc"]) & details["auc"].gt(0.5)
    )
    details["rank_ic_threshold_passed"] = (
        np.isfinite(details["rank_ic"]) & details["rank_ic"].gt(0.0)
    )
    details["auc_passed"] = (
        details["gate_eligible_fold"]
        & details["coverage_passed"]
        & details["auc_threshold_passed"]
    )
    details["rank_ic_passed"] = (
        details["gate_eligible_fold"]
        & details["coverage_passed"]
        & details["rank_ic_threshold_passed"]
    )
    details["joint_direction_passed"] = (
        details["auc_passed"] & details["rank_ic_passed"]
    )

    def failure_reason(row):
        if not row["gate_eligible_fold"]:
            if row["expected_observations"] == 0 and "model_eligibility_start" in row:
                return "pre_model_inception;excluded_from_ratio"
            return "scheduled_fold_shorter_than_minimum;excluded_from_ratio"
        reasons = []
        if not row["coverage_passed"]:
            reasons.append("insufficient_predictions")
        if not np.isfinite(row["auc"]):
            reasons.append("auc_unavailable")
        elif not row["auc_threshold_passed"]:
            reasons.append("auc_not_above_0.5")
        if not np.isfinite(row["rank_ic"]):
            reasons.append("rank_ic_unavailable")
        elif not row["rank_ic_threshold_passed"]:
            reasons.append("rank_ic_not_above_0")
        return "passed" if not reasons else "|".join(reasons)

    details["failure_reason"] = details.apply(failure_reason, axis=1)
    eligible = details.loc[details["gate_eligible_fold"]]
    eligible_count = len(eligible)
    coverage_ratio = (
        float(eligible["coverage_passed"].mean()) if eligible_count else 0.0
    )
    joint_ratio = (
        float(eligible["joint_direction_passed"].mean()) if eligible_count else 0.0
    )
    aggregate_auc_passed = bool(np.isfinite(aggregate_auc) and aggregate_auc > 0.5)
    aggregate_rank_ic_passed = bool(
        np.isfinite(aggregate_rank_ic) and aggregate_rank_ic > 0.0
    )
    fold_coverage_passed = bool(eligible_count > 0 and coverage_ratio == 1.0)
    fold_direction_passed = bool(joint_ratio >= minimum_joint_pass_ratio)
    signal_gate_passed = bool(
        aggregate_auc_passed
        and aggregate_rank_ic_passed
        and fold_coverage_passed
        and fold_direction_passed
    )
    summary = {
        "aggregate_auc": aggregate_auc,
        "aggregate_auc_threshold": 0.5,
        "aggregate_auc_passed": aggregate_auc_passed,
        "aggregate_rank_ic": aggregate_rank_ic,
        "aggregate_rank_ic_threshold": 0.0,
        "aggregate_rank_ic_passed": aggregate_rank_ic_passed,
        "configured_folds": len(details),
        "gate_eligible_folds": eligible_count,
        "fully_evaluable_folds": int(eligible["coverage_passed"].sum()),
        "fold_evaluable_ratio": coverage_ratio,
        "fold_coverage_passed": fold_coverage_passed,
        "fold_joint_direction_pass_ratio": joint_ratio,
        "minimum_fold_joint_direction_pass_ratio": minimum_joint_pass_ratio,
        "fold_direction_passed": fold_direction_passed,
        "signal_gate_passed": signal_gate_passed,
        "direction_metric_definition": (
            "share of eligible outer folds with both AUC>0.5 and Rank IC>0; "
            "not factor-coefficient sign stability"
        ),
    }
    return summary, details


def _policy_kwargs(config, policy):
    common = {
        "min_weight": config["min_weight"],
        "max_weight": config["max_weight"],
    }
    if policy == "fixed_bin":
        common.update(
            fixed_thresholds=config["fixed_thresholds"],
            fixed_weights=config["fixed_weights"],
        )
    elif policy == "expanding_percentile":
        common.update(
            percentile_breaks=config["percentile_breaks"],
            percentile_weights=config["percentile_weights"],
            percentile_min_history=config["percentile_min_history"],
        )
    elif policy == "smoothed_linear":
        common["smoothing_span"] = config.get("smoothing_span", 3)
    elif policy == "static_50_50":
        common["static_stock_weight"] = config.get("static_stock_weight", 0.50)
    return common


def build_policy_backtests(
    *,
    market_price,
    raw_ews,
    cash_yield,
    policies,
    transaction_cost_bps,
    sizing_config,
    cash_return_convention="simple_divide_12",
):
    backtests = {}
    for policy in policies:
        target = target_weight_from_ews(
            raw_ews,
            policy=policy,
            **_policy_kwargs(sizing_config, policy),
        )
        backtests[policy] = run_backtest(
            market_price=market_price,
            ews=raw_ews,
            target_stock_weight=target,
            allocation_policy=policy,
            cash_yield_annual_pct=cash_yield,
            transaction_cost_bps=transaction_cost_bps,
            cash_return_convention=cash_return_convention,
            verbose=False,
        )
    return backtests


def common_backtest_index(backtests):
    common = None
    for backtest in backtests.values():
        valid = backtest.index[
            backtest["executed_stock_weight"].notna()
            & backtest["market_return"].notna()
        ]
        common = valid if common is None else common.intersection(valid)
    if common is None or len(common) == 0:
        raise ValueError("No common tradable months across allocation policies")
    return common


def _evaluate_policy(backtest, index, policy, cost_bps):
    data = backtest.loc[index].copy()
    average_weight = float(data["executed_stock_weight"].mean())
    data["same_exposure_return"] = (
        average_weight * data["market_return"]
        + (1 - average_weight) * data["cash_return"]
    )
    data["active_return"] = data["strategy_return"] - data["same_exposure_return"]

    dynamic = performance_stats(data["strategy_return"], data["cash_return"])
    same = performance_stats(data["same_exposure_return"], data["cash_return"])
    active_std = data["active_return"].std()
    information_ratio = (
        data["active_return"].mean() / active_std * np.sqrt(12)
        if active_std > 0
        else np.nan
    )
    return {
        "policy": policy,
        "transaction_cost_bps": cost_bps,
        "Start": index.min().date().isoformat(),
        "End": index.max().date().isoformat(),
        "Months": len(index),
        "average_stock_weight": average_weight,
        "annual_turnover": float(data["turnover"].mean() * 12),
        "total_transaction_cost": float(data["transaction_cost"].sum()),
        "CAGR": dynamic["CAGR"],
        "Sharpe": dynamic["Sharpe"],
        "Sortino": dynamic["Sortino"],
        "MaxDrawdown": dynamic["MaxDrawdown"],
        "same_exposure_CAGR": same["CAGR"],
        "same_exposure_Sharpe": same["Sharpe"],
        "same_exposure_MaxDrawdown": same["MaxDrawdown"],
        "active_CAGR": dynamic["CAGR"] - same["CAGR"],
        "Sharpe_difference": dynamic["Sharpe"] - same["Sharpe"],
        "annualized_active_return": float(data["active_return"].mean() * 12),
        "tracking_error": float(active_std * np.sqrt(12)),
        "information_ratio": float(information_ratio),
        "drawdown_difference": dynamic["MaxDrawdown"] - same["MaxDrawdown"],
    }, data


def compare_position_sizing(
    *,
    market_price,
    raw_ews,
    cash_yield,
    policies,
    transaction_cost_scenarios,
    sizing_config,
    fold_labels=None,
    evaluation_start=None,
    evaluation_end=None,
    cash_return_convention="simple_divide_12",
):
    """Evaluate all policies on the same intersection for every cost case."""
    rows = []
    monthly_frames = []
    fold_frames = []

    for cost_bps in transaction_cost_scenarios:
        backtests = build_policy_backtests(
            market_price=market_price,
            raw_ews=raw_ews,
            cash_yield=cash_yield,
            policies=policies,
            transaction_cost_bps=cost_bps,
            sizing_config=sizing_config,
            cash_return_convention=cash_return_convention,
        )
        common = common_backtest_index(backtests)
        if evaluation_start is not None:
            common = common[common >= pd.Timestamp(evaluation_start)]
        if evaluation_end is not None:
            common = common[common <= pd.Timestamp(evaluation_end)]
        if len(common) == 0:
            raise ValueError("No common months remain inside the requested evaluation window")

        for policy, backtest in backtests.items():
            row, common_data = _evaluate_policy(backtest, common, policy, cost_bps)
            rows.append(row)
            common_data = common_data.assign(
                policy=policy,
                transaction_cost_bps=cost_bps,
            )
            monthly_frames.append(common_data)

            if fold_labels is None:
                fold_definitions = [
                    (fold, common[start : start + 24])
                    for fold, start in enumerate(range(0, len(common), 24), start=1)
                ]
            else:
                aligned_labels = fold_labels.reindex(common)
                fold_definitions = [
                    (int(fold), aligned_labels.index[aligned_labels.eq(fold)])
                    for fold in sorted(aligned_labels.dropna().unique())
                ]

            for fold, fold_index in fold_definitions:
                if len(fold_index) < 12:
                    continue
                fold_row, _ = _evaluate_policy(
                    backtest,
                    fold_index,
                    policy,
                    cost_bps,
                )
                fold_row["fold"] = fold
                fold_frames.append(fold_row)

    comparison = pd.DataFrame(rows)
    monthly = pd.concat(monthly_frames).sort_index()
    fold_results = pd.DataFrame(fold_frames)
    return comparison, monthly, fold_results


def _sharpe(values, risk_free):
    excess = values - risk_free
    std = np.std(excess, ddof=1)
    return np.mean(excess) / std * np.sqrt(12) if std > 0 else np.nan


def block_bootstrap_policy(
    monthly,
    *,
    samples=1000,
    block_months=3,
    random_seed=42,
):
    """Paired moving-block bootstrap against each policy's same exposure."""
    rows = []
    rng = np.random.default_rng(random_seed)
    group_columns = ["policy", "transaction_cost_bps"]

    for (policy, cost_bps), data in monthly.groupby(group_columns, sort=False):
        data = data.dropna(
            subset=["strategy_return", "same_exposure_return", "cash_return"]
        )
        n = len(data)
        if n < 12:
            continue
        strategy = data["strategy_return"].to_numpy()
        same = data["same_exposure_return"].to_numpy()
        cash = data["cash_return"].to_numpy()
        starts = np.arange(max(1, n - block_months + 1))
        active_means = np.empty(samples)
        sharpe_differences = np.empty(samples)

        for sample in range(samples):
            sampled = []
            while len(sampled) < n:
                start = int(rng.choice(starts))
                sampled.extend(range(start, min(start + block_months, n)))
            positions = np.asarray(sampled[:n])
            active_means[sample] = np.mean(strategy[positions] - same[positions]) * 12
            sharpe_differences[sample] = (
                _sharpe(strategy[positions], cash[positions])
                - _sharpe(same[positions], cash[positions])
            )

        for metric, values in (
            ("annualized_active_return", active_means),
            ("Sharpe_difference", sharpe_differences),
        ):
            rows.append(
                {
                    "policy": policy,
                    "transaction_cost_bps": cost_bps,
                    "metric": metric,
                    "samples": samples,
                    "block_months": block_months,
                    "estimate": float(np.nanmean(values)),
                    "ci_2_5": float(np.nanquantile(values, 0.025)),
                    "ci_50": float(np.nanquantile(values, 0.50)),
                    "ci_97_5": float(np.nanquantile(values, 0.975)),
                    "probability_positive": float(np.nanmean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def rolling_active_diagnostics(monthly, windows=(12, 24, 36)):
    """Compute rolling active return, tracking error and Sharpe difference."""
    frames = []
    for (policy, cost_bps), data in monthly.groupby(
        ["policy", "transaction_cost_bps"], sort=False
    ):
        data = data.sort_index()
        active = data["strategy_return"] - data["same_exposure_return"]
        strategy_excess = data["strategy_return"] - data["cash_return"]
        same_excess = data["same_exposure_return"] - data["cash_return"]
        for window in windows:
            active_mean = active.rolling(window, min_periods=window).mean() * 12
            tracking_error = active.rolling(window, min_periods=window).std() * np.sqrt(12)
            strategy_sharpe = (
                strategy_excess.rolling(window, min_periods=window).mean()
                / strategy_excess.rolling(window, min_periods=window).std()
                * np.sqrt(12)
            )
            same_sharpe = (
                same_excess.rolling(window, min_periods=window).mean()
                / same_excess.rolling(window, min_periods=window).std()
                * np.sqrt(12)
            )
            frame = pd.DataFrame(
                {
                    "date": data.index,
                    "policy": policy,
                    "transaction_cost_bps": cost_bps,
                    "window_months": window,
                    "annualized_active_return": active_mean.to_numpy(),
                    "tracking_error": tracking_error.to_numpy(),
                    "information_ratio": (active_mean / tracking_error).to_numpy(),
                    "Sharpe_difference": (strategy_sharpe - same_sharpe).to_numpy(),
                }
            ).dropna(subset=["annualized_active_return"])
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_holdout_safety_veto(
    *,
    auc,
    rank_ic,
    dynamic_sharpe,
    same_exposure_sharpe,
    dynamic_cagr,
    same_exposure_cagr,
    dynamic_max_drawdown,
    same_exposure_max_drawdown,
    maximum_drawdown_disadvantage=0.03,
):
    """Return a conservative holdout veto; a pass is never a promotion.

    The caller must lock the feature set and allocation policy before this
    evaluation.  The result may only stop tactical use, never select a better
    policy from the holdout.
    """
    differences = {
        "sharpe_difference": float(dynamic_sharpe - same_exposure_sharpe),
        "cagr_difference": float(dynamic_cagr - same_exposure_cagr),
        "drawdown_difference": float(
            dynamic_max_drawdown - same_exposure_max_drawdown
        ),
    }
    checks = {
        "auc_above_half": bool(np.isfinite(auc) and auc > 0.5),
        "rank_ic_positive": bool(np.isfinite(rank_ic) and rank_ic > 0.0),
        "sharpe_not_worse": bool(differences["sharpe_difference"] >= 0.0),
        "cagr_not_worse": bool(differences["cagr_difference"] >= 0.0),
        "drawdown_not_worse_by_more_than_3pct": bool(
            differences["drawdown_difference"] >= -maximum_drawdown_disadvantage
        ),
    }
    return bool(all(checks.values())), checks, differences


def select_position_policy(comparison, fold_results, baseline="linear"):
    """Apply the portfolio gate and fail closed to a declared static policy.

    Among policies that clear every stability gate, prefer the highest active
    return after the more conservative 25 bp transaction-cost scenario.  Fold
    Sharpe remains a hard gate and a secondary ranking key; using it as the
    primary key tended to favour coarse high-turnover bins even when a smoother
    policy delivered more development-sample alpha after costs.
    """
    decisions = []
    for policy in comparison["policy"].drop_duplicates():
        summary = comparison[
            (comparison["policy"] == policy)
            & (comparison["transaction_cost_bps"].isin([10, 25]))
        ]
        folds = fold_results[
            (fold_results["policy"] == policy)
            & (fold_results["transaction_cost_bps"] == 10)
        ]
        positive_ratio = (
            float((folds["annualized_active_return"] > 0).mean())
            if not folds.empty
            else 0.0
        )
        median_sharpe_gain = float(folds["Sharpe_difference"].median()) if not folds.empty else np.nan
        cost_positive = bool(
            len(summary) == 2 and (summary["annualized_active_return"] > 0).all()
        )
        active_return_by_cost = {
            int(row.transaction_cost_bps): float(row.annualized_active_return)
            for row in summary.itertuples(index=False)
        }
        drawdown_ok = bool(
            not summary.empty and (summary["drawdown_difference"] >= -0.03).all()
        )
        drawdown_by_cost = {
            int(row.transaction_cost_bps): float(row.drawdown_difference)
            for row in summary.itertuples(index=False)
        }
        sharpe_passed = bool(
            np.isfinite(median_sharpe_gain) and median_sharpe_gain >= 0.10
        )
        positive_fold_passed = bool(positive_ratio >= 2 / 3)
        passed = bool(
            sharpe_passed
            and positive_fold_passed
            and cost_positive
            and drawdown_ok
        )
        failed_conditions = []
        if not sharpe_passed:
            failed_conditions.append("median_fold_sharpe_gain_below_0.10")
        if not positive_fold_passed:
            failed_conditions.append("positive_fold_ratio_below_two_thirds")
        if not cost_positive:
            failed_conditions.append("active_return_not_positive_at_10_and_25_bps")
        if not drawdown_ok:
            failed_conditions.append("drawdown_difference_below_minus_0.03")
        decisions.append(
            {
                "policy": policy,
                "median_fold_Sharpe_difference": median_sharpe_gain,
                "minimum_median_fold_Sharpe_difference": 0.10,
                "median_fold_Sharpe_gate": sharpe_passed,
                "positive_fold_ratio": positive_ratio,
                "minimum_positive_fold_ratio": 2 / 3,
                "positive_fold_ratio_gate": positive_fold_passed,
                "annualized_active_return_10bps": active_return_by_cost.get(10, np.nan),
                "annualized_active_return_25bps": active_return_by_cost.get(25, np.nan),
                "cost_10_25_positive": cost_positive,
                "drawdown_difference_10bps": drawdown_by_cost.get(10, np.nan),
                "drawdown_difference_25bps": drawdown_by_cost.get(25, np.nan),
                "minimum_drawdown_difference": -0.03,
                "drawdown_gate": drawdown_ok,
                "portfolio_gate_passed": passed,
                "policy_kind": (
                    "static_fail_closed_fallback"
                    if policy == "static_50_50"
                    else "tactical_candidate"
                ),
                "failed_conditions": "|".join(failed_conditions) if failed_conditions else "passed",
            }
        )
    decision_table = pd.DataFrame(decisions)
    eligible = decision_table[
        decision_table["portfolio_gate_passed"]
        & ~decision_table["policy"].eq("static_50_50")
    ]
    if eligible.empty:
        if baseline not in set(decision_table["policy"]):
            raise ValueError(f"Fallback policy {baseline!r} was not evaluated")
        winner = str(baseline)
        reason = "no_tactical_policy_passed;fail_closed_fallback"
    else:
        winner = str(
            eligible.sort_values(
                [
                    "annualized_active_return_25bps",
                    "median_fold_Sharpe_difference",
                    "positive_fold_ratio",
                    "drawdown_difference_25bps",
                    "policy",
                ],
                ascending=False,
            ).iloc[0]["policy"]
        )
        reason = (
            "pre_holdout_portfolio_gate_passed;"
            "max_active_return_at_25bps_then_fold_sharpe"
        )
    decision_table["selected"] = decision_table["policy"].eq(winner)
    decision_table["selected_as_fallback"] = (
        decision_table["selected"]
        & decision_table["policy"].eq(baseline)
        & eligible.empty
    )
    decision_table["selection_reason"] = np.where(
        decision_table["selected"], reason, "not_selected"
    )
    return winner, decision_table


def enforce_signal_gate_fallback(
    selected_policy,
    decision_table,
    *,
    signal_gate_passed,
    fallback="static_50_50",
):
    """Disable tactical allocation when its underlying signal gate fails."""
    result = decision_table.copy()
    if signal_gate_passed:
        return selected_policy, result
    if fallback not in set(result["policy"]):
        raise ValueError(f"Signal-gate fallback policy {fallback!r} was not evaluated")
    result["selected"] = result["policy"].eq(fallback)
    result["selected_as_fallback"] = result["selected"]
    result["selection_reason"] = np.where(
        result["selected"],
        "signal_gate_failed;fail_closed_fallback",
        "not_selected",
    )
    return fallback, result
