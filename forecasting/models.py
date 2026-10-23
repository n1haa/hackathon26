"""Weekday baseline, weekday exponential smoothing, recursive gradient boosting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .features import build_features


MODEL_NAMES = ("weekday_baseline", "exponential_smoothing", "hist_gradient_boosting")


def predict_flows(history, dates, model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    if history.empty or history[["income", "expense"]].isna().any().any():
        raise ValueError("Models require nonempty observed daily income and expense")
    if model_name == "hist_gradient_boosting":
        values = _ml(history, dates)
    else:
        values = _weekday(history, dates, smoothing=model_name == "exponential_smoothing")
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} produced nonfinite predictions")
    return pd.DataFrame(np.maximum(values, 0), index=dates, columns=["income", "expense"])


def _weekday(history, dates, smoothing):
    means = history[["income", "expense"]].mean().to_numpy()
    profiles = {}
    for weekday, rows in history.groupby(history.index.dayofweek):
        values = rows[["income", "expense"]]
        profiles[weekday] = (values.ewm(alpha=0.3, adjust=False).mean().iloc[-1].to_numpy()
                             if smoothing else values.mean().to_numpy())
    return np.array([profiles.get(d.dayofweek, means) for d in dates])


def _ml(history, dates):
    # Do not learn a category vocabulary from columns that only appear after
    # this training cutoff (daily aggregation may have used a longer statement).
    history = history.drop(columns=[c for c in history if c.startswith("category::") and not history[c].any()])
    x = build_features(history)
    valid = x.notna().all(axis=1)
    if valid.sum() < 28:
        raise ValueError("hist_gradient_boosting requires at least 56 complete calendar days")
    estimators = []
    with threadpool_limits(limits=1):
        for target in ("income", "expense"):
            estimator = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
                        min_samples_leaf=10, learning_rate=0.07, l2_regularization=1,
                        early_stopping=False, random_state=42)
            estimator.fit(x.loc[valid], history.loc[valid, target])
            estimators.append(estimator)
        category_columns = [c for c in history if c.startswith("category::")]
        shares = {}
        for column in category_columns:
            target = "income" if column.split("::")[1] == "incoming" else "expense"
            denominator = history[target].sum()
            shares[column] = (target, history[column].sum() / denominator if denominator else 0.)
        working = history[["income", "expense"] + category_columns].copy()
        predictions = []
        for date in dates:
            working.loc[date] = np.nan
            row = build_features(working).iloc[[-1]].reindex(columns=x.columns)
            pair = [max(0., float(estimator.predict(row)[0])) for estimator in estimators]
            working.loc[date, ["income", "expense"]] = pair
            for column, (target, share) in shares.items():
                working.loc[date, column] = working.loc[date, target] * share
            predictions.append(pair)
    return np.asarray(predictions)
