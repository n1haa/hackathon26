"""Backend/risk-engine entry point with JSON-compatible output."""
from datetime import datetime, timezone as datetime_timezone
from decimal import Decimal
import warnings

import pandas as pd

from .data import money, prepare_daily_cashflows, rounded_prediction
from .evaluation import evaluate_models, positive_integer, validate_daily
from .events import separate_events
from .models import MODEL_NAMES, predict_flows


class ForecastWarning(UserWarning):
    """A statement inconsistency affecting forecast interpretation."""


def forecast_cashflow(transactions, current_balance, horizon=30, *,
                      as_of=None, history_start=None, complete_history=False,
                      observed_dates=None, opening_balance=None, recurring_events=None,
                      fields=None, timezone="Europe/Moscow", model="auto"):
    """Forecast from end-of-day as_of; balance is supplied at that same cutoff.

    Without as_of, cutoff is the last transaction's date, not the system date.
    Unknown statement days cause an error instead of silently becoming zero.
    """
    horizon = positive_integer(horizon, "horizon")
    balance = money(current_balance, "current_balance")
    if model != "auto" and model not in MODEL_NAMES:
        raise ValueError(f"model must be auto or one of {MODEL_NAMES}")
    prepared = prepare_daily_cashflows(transactions, as_of=as_of, history_start=history_start,
                 complete_history=complete_history, observed_dates=observed_dates,
                 opening_balance=opening_balance, current_balance=current_balance,
                 timezone=timezone, fields=fields)
    validate_daily(prepared.daily)
    daily, events, notes = separate_events(prepared, recurring_events, horizon, timezone)
    report = evaluate_models(daily, horizon)
    report["scope"] = "residual_flows_without_recurring_series" if events else "historical_flows"
    selected = report["selected_model"] if model == "auto" else model
    messages = list(prepared.warnings) + notes
    if report["status"] == "insufficient_history":
        messages.append(report["reason"] + "; automatic selection uses weekday_baseline, quality is unmeasured.")
    if len(daily) < 7:
        messages.append("Less than one week of history: unseen weekdays use the overall historical mean.")
    if model != "auto":
        messages.append("Model was explicitly requested; validation-selected model is recorded separately in metrics.")
    dates = pd.date_range(prepared.as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    try:
        predicted = predict_flows(daily, dates, selected)
    except (ValueError, FloatingPointError) as exc:
        if model != "auto" or selected == MODEL_NAMES[0]:
            raise
        messages.append(f"{selected} failed during final fit: {exc}; using weekday_baseline.")
        selected = MODEL_NAMES[0]
        predicted = predict_flows(daily, dates, selected)
    scheduled = {}
    for event in events:
        key = (event["date"], event["direction"])
        scheduled[key] = scheduled.get(key, Decimal(0)) + event["amount"]
    records = []
    for date, row in predicted.iterrows():
        date_string = date.date().isoformat()
        income_model = rounded_prediction(row["income"])
        expense_model = rounded_prediction(row["expense"])
        income_event = scheduled.get((date_string, "incoming"), Decimal(0))
        expense_event = scheduled.get((date_string, "outgoing"), Decimal(0))
        income, expense = income_model + income_event, expense_model + expense_event
        net = income - expense
        balance += net
        records.append({"date": date_string, "predicted_income": float(income),
                        "predicted_expense": float(expense), "predicted_net_flow": float(net),
                        "predicted_balance": float(balance),
                        "model_income": float(income_model), "model_expense": float(expense_model),
                        "scheduled_income": float(income_event), "scheduled_expense": float(expense_event)})
    for message in messages:
        if "mismatch" in message.lower():
            warnings.warn(message, ForecastWarning, stacklevel=2)
    return {"model_name": selected, "horizon": horizon,
            "generated_at": datetime.now(datetime_timezone.utc).isoformat(),
            "as_of": prepared.as_of.date().isoformat(), "timezone": timezone,
            "current_balance": float(money(current_balance)),
            "history_days": len(daily), "forecast": records,
            "metrics": report, "warnings": messages,
            "recurring_events": [{**e, "amount": float(e["amount"])} for e in events]}
