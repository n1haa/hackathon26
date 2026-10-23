"""Thin integration layer for recurring_detector and risk_engine.

This module deliberately does not implement forecasting. It prepares recurring
payment signals for an existing forecasting module, then evaluates that
module's resulting projected balances with risk_engine.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from recurring_detector import detect_recurring
from risk_engine import RiskThresholds, analyze_cash_gap

DATE_ALIASES = ("date", "forecast_date", "operation_date", "ds", "day")
TRANSACTION_DATE_ALIASES = ("operation_date", "date", "transaction_date")
COMPANY_ID_ALIASES = ("company_id",)

ForecastBuilder = Callable[..., Any]


def _find_column(columns: Iterable[Any], aliases: Iterable[str]) -> Any | None:
    lookup = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except (TypeError, ValueError, OverflowError):
        return None


def _to_transaction_frame(transactions: Any) -> pd.DataFrame:
    if isinstance(transactions, pd.DataFrame):
        return transactions.copy()
    if isinstance(transactions, (list, tuple)):
        return pd.DataFrame(transactions)
    raise TypeError("transactions должен быть pandas.DataFrame или списком словарей")


def _validate_single_company(transactions: pd.DataFrame) -> str | None:
    """A single forecast must never combine different companies' balances."""
    company_col = _find_column(transactions.columns, COMPANY_ID_ALIASES)
    if company_col is None:
        return None

    values = [
        str(value).strip()
        for value in transactions[company_col].tolist()
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan"
    ]
    unique_ids = list(dict.fromkeys(values))
    if len(unique_ids) > 1:
        raise ValueError(
            "run_cashflow_pipeline обрабатывает одну компанию за вызов: "
            f"найдено несколько company_id: {unique_ids[:10]}. "
            "Разделите операции по компании и передайте соответствующий прогноз."
        )
    return unique_ids[0] if unique_ids else None


def _forecast_rows(forecast: Any) -> pd.DataFrame | None:
    if forecast is None:
        return None
    if isinstance(forecast, pd.DataFrame):
        return forecast.copy()
    if isinstance(forecast, Mapping):
        payload = None
        for key in ("forecast", "predictions", "data", "items", "results"):
            if key in forecast:
                payload = forecast[key]
                break
        if payload is None:
            payload = [forecast]
        try:
            return pd.DataFrame(payload)
        except (TypeError, ValueError):
            return None
    if isinstance(forecast, (list, tuple)):
        return pd.DataFrame(forecast)
    return None


def _forecast_bounds(forecast: Any) -> tuple[date, date] | None:
    frame = _forecast_rows(forecast)
    if frame is None or frame.empty:
        return None
    date_col = _find_column(frame.columns, DATE_ALIASES)
    if date_col is None:
        return None
    valid = [_as_date(value) for value in frame[date_col].tolist()]
    valid = [value for value in valid if value is not None]
    if not valid:
        return None
    return min(valid), max(valid)


def _history_end_date(transactions: pd.DataFrame) -> date | None:
    date_col = _find_column(transactions.columns, TRANSACTION_DATE_ALIASES)
    if date_col is None:
        return None
    values = [_as_date(value) for value in transactions[date_col].tolist()]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def prepare_recurring_events_for_forecast(
    recurring_events: list[dict[str, Any]],
    *,
    start_date: str | date,
    horizon_days: int = 30,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Convert detector output to a compact, dated forecasting input payload.

    Only the detector's `expected_next_date` is used. This function does not
    invent additional occurrences or extrapolate a recurring schedule. Events
    outside the forecast window are excluded (including stale expected dates).
    """
    start = _as_date(start_date)
    if start is None:
        raise ValueError(f"Некорректная start_date: {start_date!r}")
    if horizon_days < 1:
        raise ValueError("horizon_days должен быть >= 1")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence должен быть в диапазоне [0, 1]")

    end = start + timedelta(days=horizon_days - 1)
    selected: list[dict[str, Any]] = []
    excluded_outside_window = 0
    excluded_low_confidence = 0
    warnings: list[str] = []

    for index, event in enumerate(recurring_events, start=1):
        try:
            confidence = float(event.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence) or confidence < min_confidence:
            excluded_low_confidence += 1
            continue

        expected_date = _as_date(event.get("expected_next_date"))
        if expected_date is None or expected_date < start or expected_date > end:
            excluded_outside_window += 1
            continue

        direction_raw = str(event.get("direction", "")).strip().lower()
        if direction_raw in {"поступление", "income", "inflow", "приход", "credit"}:
            direction = "Поступление"
            sign = 1.0
        elif direction_raw in {"списание", "expense", "outflow", "расход", "debit"}:
            direction = "Списание"
            sign = -1.0
        else:
            warnings.append(
                f"Событие {index} пропущено: неизвестное направление "
                f"{event.get('direction')!r}."
            )
            continue

        raw_amount = event.get("expected_amount", event.get("median_amount"))
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            warnings.append(f"Событие {index} пропущено: некорректная сумма.")
            continue
        if not math.isfinite(amount) or amount < 0:
            warnings.append(f"Событие {index} пропущено: сумма должна быть конечной и >= 0.")
            continue

        selected.append(
            {
                "event_id": f"recurring-{len(selected) + 1:03d}",
                "company_id": event.get("company_id"),
                "date": expected_date.isoformat(),
                "counterparty": event.get("counterparty"),
                "category": event.get("category"),
                "direction": direction,
                "amount": round(amount, 2),
                # Forecasting can apply this signed flow to its daily flow model.
                "signed_amount": round(sign * amount, 2),
                "frequency": event.get("frequency"),
                "interval_days": event.get("interval_days"),
                "occurrences_count": event.get("occurrences_count"),
                "confidence": round(confidence, 3),
                "source": "recurring_detector",
            }
        )

    if excluded_outside_window:
        warnings.append(
            f"Из {len(recurring_events)} обнаруженных шаблонов {excluded_outside_window} "
            f"не имеют ожидаемой даты в окне {start.isoformat()}—{end.isoformat()}; "
            "они не переданы в forecasting. В частности, просроченные ожидаемые даты "
            "не сдвигались автоматически на будущее."
        )
    if excluded_low_confidence:
        warnings.append(
            f"Не передано шаблонов с confidence ниже {min_confidence:.2f}: "
            f"{excluded_low_confidence}. Confidence — оценка регулярности, не вероятность платежа."
        )

    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "horizon_days": horizon_days,
        "events": selected,
        "excluded_outside_window": excluded_outside_window,
        "excluded_low_confidence": excluded_low_confidence,
        "warnings": warnings,
    }


def run_cashflow_pipeline(
    transactions: Any,
    forecast: Any = None,
    *,
    forecast_builder: ForecastBuilder | None = None,
    forecast_start_date: str | date | None = None,
    horizon_days: int | None = None,
    min_occurrences: int = 3,
    min_confidence: float = 0.55,
    thresholds: RiskThresholds | None = None,
) -> dict[str, Any]:
    """Connect recurrence detection with cash-gap analysis for one company.

    Parameters
    ----------
    transactions:
        Historical transaction table for one company (e.g. C002.csv).
    forecast:
        Existing balance forecast, in the schema accepted by risk_engine.
        May be omitted only when forecast_builder is supplied.
    forecast_builder:
        Optional adapter owned by the forecasting module. It is called as
        `forecast_builder(transactions=..., recurring_events=..., base_forecast=...,
                          start_date=..., horizon_days=...)` and must return a
        forecast containing daily forecast balances. It is responsible for
        actually incorporating the supplied recurring events; this module does
        not calculate or alter projected balances itself.
    forecast_start_date:
        First date in the forecast window. Otherwise inferred from the existing
        forecast, or from the day after the last historical transaction.
    horizon_days:
        Window length. If omitted, inferred from forecast dates; otherwise 30.

    Returns recurring events, the compact event payload for forecasting, the
    cash-gap analysis and integration warnings.
    """
    tx = _to_transaction_frame(transactions)
    company_id = _validate_single_company(tx)
    recurring_events = detect_recurring(
        tx,
        min_occurrences=min_occurrences,
        min_confidence=min_confidence,
    )

    bounds = _forecast_bounds(forecast)
    start = _as_date(forecast_start_date) if forecast_start_date is not None else None
    if forecast_start_date is not None and start is None:
        raise ValueError(f"Некорректная forecast_start_date: {forecast_start_date!r}")
    if start is None and bounds is not None:
        start = bounds[0]
    if start is None:
        history_end = _history_end_date(tx)
        if history_end is not None:
            start = history_end + timedelta(days=1)
        else:
            raise ValueError(
                "Невозможно определить начало прогноза: укажите forecast_start_date "
                "или передайте существующий forecast с датами."
            )

    if horizon_days is None:
        if bounds is not None and forecast_start_date is None:
            horizon_days = (bounds[1] - bounds[0]).days + 1
        else:
            horizon_days = 30
    if horizon_days < 1:
        raise ValueError("horizon_days должен быть >= 1")

    prepared = prepare_recurring_events_for_forecast(
        recurring_events,
        start_date=start,
        horizon_days=horizon_days,
        min_confidence=min_confidence,
    )
    warnings = list(prepared["warnings"])

    final_forecast = forecast
    if forecast_builder is not None:
        final_forecast = forecast_builder(
            transactions=tx.copy(),
            recurring_events=prepared["events"],
            base_forecast=forecast,
            start_date=prepared["start_date"],
            horizon_days=horizon_days,
        )
        if final_forecast is None:
            raise ValueError("forecast_builder вернул None вместо прогноза")
    elif forecast is None:
        raise ValueError(
            "Нужно передать готовый forecast или forecast_builder. "
            "Модуль не генерирует прогноз самостоятельно."
        )
    else:
        warnings.append(
            "forecast_builder не передан: кассовый риск рассчитан по готовому "
            "прогнозу без изменения. Подготовленные recurring events не были "
            "встроены в этот прогноз; передайте их в forecasting до оценки риска."
        )

    cash_gap = analyze_cash_gap(final_forecast, thresholds=thresholds)
    warnings.extend(cash_gap.get("warnings", []))

    return {
        "company_id": company_id,
        "recurring_events": recurring_events,
        "forecast_inputs": prepared["events"],
        "forecast_window": {
            "start_date": prepared["start_date"],
            "end_date": prepared["end_date"],
            "horizon_days": horizon_days,
        },
        "forecast_builder_used": forecast_builder is not None,
        "cash_gap": cash_gap,
        # Convenient top-level aliases for API/UI callers.
        "status": cash_gap["status"],
        "minimum_balance": cash_gap["minimum_balance"],
        "maximum_deficit": cash_gap["maximum_deficit"],
        "probability": cash_gap["probability"],
        "probability_status": cash_gap["probability_status"],
        "warnings": list(dict.fromkeys(warnings)),
    }
