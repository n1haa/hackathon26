"""Thin integration layer for recurring_detector and risk_engine.

This module deliberately does not implement forecasting. It prepares recurring
payment signals for an existing forecasting module, then evaluates that
module's resulting projected balances with risk_engine.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import math
from numbers import Integral
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

if __package__:
    from .recurring_detector import detect_recurring
    from .risk_engine import BALANCE_ALIASES, RiskThresholds, analyze_cash_gap
    from .data_loader import validate_transactions
else:  # Support the existing `python cashflow_integrated/main.py` entry point.
    from recurring_detector import detect_recurring
    from risk_engine import BALANCE_ALIASES, RiskThresholds, analyze_cash_gap
    from data_loader import validate_transactions

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
    if value is None or isinstance(value, (int, float, bool)):
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

    if transactions[company_col].isna().any() or transactions[company_col].astype(str).str.strip().eq("").any():
        raise ValueError("company_id не должен быть пустым, если столбец передан")
    values = [str(value).strip() for value in transactions[company_col].tolist()]
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
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, Integral) or horizon_days < 1:
        raise ValueError("horizon_days должен быть целым числом >= 1")
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
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or confidence < min_confidence:
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


def prepare_forecast_inputs(
    transactions: Any,
    *,
    start_date: str | date | None = None,
    horizon_days: int = 30,
    min_occurrences: int = 3,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Prepare events without requiring an already computed forecast.

    History must end before the forecast starts, including during backtesting.
    Only one next occurrence per detected pattern is provided; this is not a
    complete payment schedule. Forecasting owns the decision to extrapolate it.
    """
    tx = validate_transactions(_to_transaction_frame(transactions))
    if isinstance(min_occurrences, bool) or not isinstance(min_occurrences, Integral) or min_occurrences < 3:
        raise ValueError("min_occurrences должен быть целым числом >= 3")
    company_id = _validate_single_company(tx)
    history_end = _history_end_date(tx)
    start = _as_date(start_date) if start_date is not None else (
        history_end + timedelta(days=1) if history_end else None
    )
    if start is None:
        raise ValueError("Укажите корректную start_date или непустую историю операций")
    if history_end is not None and history_end >= start:
        raise ValueError("История должна заканчиваться до start_date: утечка будущих данных")
    # Validate the window before running the detector.
    prepare_recurring_events_for_forecast([], start_date=start, horizon_days=horizon_days,
                                          min_confidence=min_confidence)
    events = detect_recurring(tx, min_occurrences=min_occurrences, min_confidence=min_confidence)
    prepared = prepare_recurring_events_for_forecast(
        events, start_date=start, horizon_days=horizon_days, min_confidence=min_confidence,
    )
    return {**prepared, "company_id": company_id, "recurring_events": events}


def _forecast_for_window(forecast: Any, *, company_id: str | None,
                         start: date, horizon_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate ownership and expose missing boundary days to the risk caller."""
    frame = _forecast_rows(forecast)
    if frame is None:
        raise ValueError("Некорректный контейнер forecast")
    forecast_company = _validate_single_company(frame)
    envelope_company = forecast.get("company_id") if isinstance(forecast, Mapping) else None
    for owner in (forecast_company, envelope_company):
        if owner is not None and company_id is not None and str(owner).strip() != company_id:
            raise ValueError("company_id прогноза не совпадает с компанией операций")
    end = start + timedelta(days=int(horizon_days) - 1)
    date_col = _find_column(frame.columns, DATE_ALIASES)
    balance_col = _find_column(frame.columns, BALANCE_ALIASES)
    expected = {start + timedelta(days=i) for i in range(int(horizon_days))}
    valid_dates: list[date] = []
    if date_col is not None:
        dates = frame[date_col].map(_as_date)
        if any(day is not None and not start <= day <= end for day in dates):
            raise ValueError("Даты forecast выходят за выбранное окно прогноза")
        if balance_col is not None:
            for day, value in zip(dates, frame[balance_col]):
                try:
                    if day is not None and math.isfinite(float(value)):
                        valid_dates.append(day)
                except (TypeError, ValueError, OverflowError):
                    pass
    missing = sorted(expected - set(valid_dates))
    complete = len(frame) == horizon_days and len(valid_dates) == len(set(valid_dates)) and not missing
    return frame, {
        "complete": complete,
        "valid_days": len(set(valid_dates)),
        "expected_days": int(horizon_days),
        "missing_dates": [day.isoformat() for day in missing],
    }


def run_cashflow_pipeline(
    transactions: Any,
    forecast: Any = None,
    *,
    forecast_builder: ForecastBuilder | None = None,
    forecast_start_date: str | date | None = None,
    horizon_days: int = 30,
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
        Window length; defaults to 30 even if the supplied forecast is shorter.

    Returns recurring events, the compact event payload for forecasting, the
    cash-gap analysis and integration warnings.
    """
    tx = validate_transactions(_to_transaction_frame(transactions))
    company_id = _validate_single_company(tx)

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

    prepared = prepare_forecast_inputs(
        tx,
        start_date=start,
        horizon_days=horizon_days,
        min_occurrences=min_occurrences,
        min_confidence=min_confidence,
    )
    recurring_events = prepared["recurring_events"]
    warnings = list(prepared["warnings"])

    final_forecast = forecast
    if forecast_builder is not None:
        final_forecast = forecast_builder(
            transactions=tx.copy(),
            recurring_events=deepcopy(prepared["events"]),
            base_forecast=deepcopy(forecast),
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

    final_frame, coverage = _forecast_for_window(
        final_forecast, company_id=company_id, start=start, horizon_days=horizon_days,
    )
    cash_gap = analyze_cash_gap(final_frame, thresholds=thresholds)
    if not coverage["complete"]:
        cash_gap["warnings"].append(
            f"Прогноз покрывает {coverage['valid_days']} из {horizon_days} дней окна "
            "или содержит некорректные/повторные строки; полнота не подтверждена."
        )
        if cash_gap["status"] == "no_gap":
            cash_gap["status"] = "incomplete"
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
        "forecast_coverage": coverage,
        "cash_gap": cash_gap,
        # Convenient top-level aliases for API/UI callers.
        "status": cash_gap["status"],
        "minimum_balance": cash_gap["minimum_balance"],
        "maximum_deficit": cash_gap["maximum_deficit"],
        "probability": cash_gap["probability"],
        "probability_status": cash_gap["probability_status"],
        "warnings": list(dict.fromkeys(warnings)),
    }
