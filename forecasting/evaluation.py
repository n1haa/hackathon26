"""Expanding chronological validation; terminal test never selects the model."""
import numpy as np
import pandas as pd

from .models import MODEL_NAMES, predict_flows
from .data import rounded_prediction


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_daily(daily):
    if not isinstance(daily.index, pd.DatetimeIndex) or not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Expected sorted unique daily DatetimeIndex")
    if len(daily) > 1 and not (daily.index[1:] - daily.index[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("Expected continuous daily calendar")
    if daily.empty or not {"income", "expense"}.issubset(daily):
        raise ValueError("Nonempty daily income and expense are required")
    values = daily[["income", "expense"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Unknown/missing coverage or invalid daily flows; provide explicit complete statement coverage")
    if "observed" in daily and not daily["observed"].eq(True).all():
        raise ValueError("Unknown/missing coverage; provide explicit complete statement coverage")


def _flow_metrics(actual, predicted):
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "wape": float(np.abs(error).sum() / denominator) if denominator else None}


def _rounded_flows(predicted):
    return predicted.apply(lambda column: column.map(lambda value: float(rounded_prediction(value))))


def calculate_metrics(actual, predicted, *, balance_errors=None):
    a = actual[["income", "expense"]].to_numpy(dtype=float)
    p = predicted[["income", "expense"]].to_numpy(dtype=float)
    income = _flow_metrics(a[:, 0], p[:, 0])
    expense = _flow_metrics(a[:, 1], p[:, 1])
    if balance_errors is None:
        balance_errors = np.cumsum((p[:, 0] - p[:, 1]) - (a[:, 0] - a[:, 1]))
    return {"income": income, "expense": expense,
            "flow_mae": (income["mae"] + expense["mae"]) / 2,
            "balance_mae": float(np.abs(balance_errors).mean())}


def evaluate_models(daily, horizon=30, *, min_train_days=56, max_folds=3):
    """Train | validation folds | test, all with full recursive horizon.

    Selection minimizes mean income/expense MAE on validation. Ties retain the
    weekday baseline. Balance errors reset to the known starting balance each fold.
    """
    horizon = positive_integer(horizon, "horizon")
    min_train_days = max(56, positive_integer(min_train_days, "min_train_days"))
    max_folds = positive_integer(max_folds, "max_folds")
    validate_daily(daily)
    report = {"status": "insufficient_history", "selected_model": MODEL_NAMES[0],
              "selection_metric": "validation.flow_mae", "horizon": horizon,
              "validation": {}, "test": {}, "folds": [], "failures": {}}
    validation_start = len(daily) - 2 * horizon
    test_start = len(daily) - horizon
    if validation_start < min_train_days:
        report["reason"] = f"Need {min_train_days + 2 * horizon} complete days for train/validation/test; got {len(daily)}"
        return report
    starts = [validation_start - k * horizon for k in range(max_folds)
              if validation_start - k * horizon >= min_train_days][::-1]
    actuals = []
    predictions = {name: [] for name in MODEL_NAMES}
    balance_errors = {name: [] for name in MODEL_NAMES}
    for split in starts:
        train, actual = daily.iloc[:split], daily.iloc[split:split + horizon]
        fold = {"train_start": train.index[0].date().isoformat(),
                "train_end": train.index[-1].date().isoformat(),
                "validation_start": actual.index[0].date().isoformat(),
                "validation_end": actual.index[-1].date().isoformat(), "metrics": {}}
        actuals.append(actual)
        for name in MODEL_NAMES:
            if name in report["failures"]:
                continue
            try:
                predicted = _rounded_flows(predict_flows(train, actual.index, name))
                fold["metrics"][name] = calculate_metrics(actual, predicted)
                predictions[name].append(predicted)
                errors = (predicted["income"] - predicted["expense"] - actual["income"] + actual["expense"]).cumsum()
                balance_errors[name].append(errors.to_numpy())
            except (ValueError, FloatingPointError) as exc:
                report["failures"][name] = str(exc)
        report["folds"].append(fold)
    combined_actual = pd.concat(actuals)
    for name in MODEL_NAMES:
        if name not in report["failures"]:
            report["validation"][name] = calculate_metrics(combined_actual, pd.concat(predictions[name]),
                                               balance_errors=np.concatenate(balance_errors[name]))
    selected = MODEL_NAMES[0]
    for name in MODEL_NAMES[1:]:
        if name in report["validation"] and report["validation"][name]["flow_mae"] < report["validation"][selected]["flow_mae"] - 1e-9:
            selected = name
    report["selected_model"] = selected
    # Selection is complete before looking at the terminal holdout.
    train, actual = daily.iloc[:test_start], daily.iloc[test_start:]
    report["test_period"] = {"train_end": train.index[-1].date().isoformat(),
                             "start": actual.index[0].date().isoformat(),
                             "end": actual.index[-1].date().isoformat()}
    for name in MODEL_NAMES:
        if name not in report["failures"]:
            try:
                predicted = _rounded_flows(predict_flows(train, actual.index, name))
                report["test"][name] = calculate_metrics(actual, predicted)
            except (ValueError, FloatingPointError) as exc:
                report["failures"][f"test::{name}"] = str(exc)
    report["status"] = "evaluated"
    return report
