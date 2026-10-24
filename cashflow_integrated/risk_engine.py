"""CashFlow AI risk engine: detect cash gaps from an existing balance forecast.

This module does not forecast cash flows or reconstruct balances from transactions.
It consumes a forecast that already contains one projected balance per date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping
import math
import re

import pandas as pd


DATE_ALIASES = ("date", "forecast_date", "operation_date", "ds", "day")
BALANCE_ALIASES = (
    "forecast_balance",
    "predicted_balance",
    "projected_balance",
    "balance_forecast",
    "balance",
    "forecasted_balance",
)
CURRENT_BALANCE_ALIASES = (
    "current_balance",
    "starting_balance",
    "initial_balance",
    "current_cash_balance",
)


@dataclass(frozen=True)
class RiskThresholds:
    """Preliminary, configurable thresholds; not universal business rules.

    If current_balance > 0, the deficit-to-current-balance ratio helps
    classify severity. Duration thresholds remain available as a fallback.
    """
    medium_deficit_ratio: float = 0.10
    high_deficit_ratio: float = 0.30
    medium_duration_days: int = 2
    high_duration_days: int = 5

    def __post_init__(self) -> None:
        if not (0 <= self.medium_deficit_ratio < self.high_deficit_ratio):
            raise ValueError(
                "Expected 0 <= medium_deficit_ratio < high_deficit_ratio"
            )
        if self.medium_duration_days < 1 or self.high_duration_days < self.medium_duration_days:
            raise ValueError("Invalid duration thresholds")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _find_column(columns: Iterable[Any], aliases: Iterable[str]) -> Any | None:
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _as_date(value: Any) -> date:
    if _is_missing(value) or isinstance(value, (int, float, bool)):
        raise ValueError(f"Invalid forecast date: {value!r}")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid forecast date: {value!r}")
    return parsed.date()


def _to_frame(forecast: Any) -> tuple[pd.DataFrame, list[str]]:
    """Normalize supported forecast containers without deriving balances."""
    warnings: list[str] = []

    if isinstance(forecast, pd.DataFrame):
        frame = forecast.copy()
    elif isinstance(forecast, Mapping):
        # Common envelopes used by APIs and forecasting modules.
        payload = None
        for key in ("forecast", "predictions", "data", "items", "results"):
            if key in forecast:
                payload = forecast[key]
                break
        if payload is None:
            # A single row-like mapping is also accepted.
            payload = [forecast]
        frame = pd.DataFrame(payload)
    elif isinstance(forecast, (list, tuple)):
        frame = pd.DataFrame(forecast)
    else:
        raise TypeError(
            "forecast must be a pandas DataFrame, a list of row mappings, "
            "or a mapping containing forecast/predictions/data/items/results"
        )

    return frame, warnings


def _risk_level(
    maximum_deficit: float,
    duration_days: int,
    current_balance: float | None,
    thresholds: RiskThresholds,
) -> str:
    """Rule-based severity, not a probability estimate."""
    ratio = None
    if current_balance is not None and current_balance > 0:
        ratio = maximum_deficit / current_balance

    # Severe duration is a high-risk signal even without a baseline balance.
    if duration_days >= thresholds.high_duration_days:
        return "high"
    if ratio is not None and ratio >= thresholds.high_deficit_ratio:
        return "high"

    if duration_days >= thresholds.medium_duration_days:
        return "medium"
    if ratio is not None and ratio >= thresholds.medium_deficit_ratio:
        return "medium"

    return "low"


def _explanation(
    start: date,
    end: date,
    minimum_balance: float,
    maximum_deficit: float,
    duration_days: int,
    current_balance: float | None,
    risk_level: str,
) -> str:
    text = (
        f"Прогнозируемый остаток отрицателен с {start.isoformat()} "
        f"по {end.isoformat()}: минимум {minimum_balance:,.2f}, "
        f"дефицит до {maximum_deficit:,.2f}; "
        f"длительность — {duration_days} дн."
    )
    if current_balance is not None and current_balance > 0:
        ratio = maximum_deficit / current_balance * 100
        text += f" Дефицит составляет {ratio:.1f}% текущего остатка."
    text += f" Предварительный уровень риска: {risk_level}."
    return text


def detect_cash_gaps(
    forecast: Any,
    *,
    thresholds: RiskThresholds | None = None,
) -> list[dict[str, Any]]:
    """Detect contiguous calendar-day periods with forecast_balance < 0.

    Supported input:
      * DataFrame with date and forecast_balance columns;
      * list of mappings;
      * mapping with a `forecast`, `predictions`, `data`, `items`, or
        `results` array;
      * single row mapping.

    Missing calendar dates are not filled with zero. A missing date creates
    a warning in `analyze_cash_gap`; negative runs separated by a missing
    date are kept as separate events.
    """
    # Kept as a convenience for callers who only need the events list.
    analysis = _analyze(forecast, thresholds=thresholds)
    return analysis["events"]


def _analyze(
    forecast: Any,
    *,
    thresholds: RiskThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RiskThresholds()
    frame, warnings = _to_frame(forecast)

    empty_result = {
        "events": [],
        "status": "unknown",
        "minimum_balance": None,
        "maximum_deficit": 0.0,
        "warnings": warnings,
        "probability": None,
        "probability_status": "not_estimated",
    }

    if frame.empty:
        warnings.append("Прогноз пуст: недостаточно данных для оценки риска.")
        return empty_result

    date_col = _find_column(frame.columns, DATE_ALIASES)
    balance_col = _find_column(frame.columns, BALANCE_ALIASES)
    current_col = _find_column(frame.columns, CURRENT_BALANCE_ALIASES)

    if date_col is None:
        warnings.append("Не найден столбец даты прогноза.")
        return empty_result
    if balance_col is None:
        warnings.append(
            "Не найден прогнозируемый остаток. Денежные потоки не "
            "пересчитывались; события кассовых разрывов не сформированы."
        )
        return empty_result

    normalized: list[dict[str, Any]] = []
    invalid_rows = 0
    for _, row in frame.iterrows():
        try:
            day = _as_date(row[date_col])
            raw_balance = row[balance_col]
            if _is_missing(raw_balance):
                raise ValueError("missing balance")
            balance = float(raw_balance)
            if not math.isfinite(balance):
                raise ValueError("non-finite balance")
            normalized.append({"date": day, "balance": balance})
        except (ValueError, TypeError, OverflowError):
            invalid_rows += 1

    if invalid_rows:
        warnings.append(
            f"Пропущено некорректных строк прогноза: {invalid_rows}."
        )

    if not normalized:
        warnings.append("Нет корректных пар дата/остаток для анализа.")
        return empty_result

    normalized.sort(key=lambda item: item["date"])
    dates = [item["date"] for item in normalized]

    # Duplicate dates make the projected balance ambiguous; don't pick one
    # arbitrarily and risk producing misleading alerts.
    duplicate_dates = sorted(
        {day for day in dates if dates.count(day) > 1}
    )
    if duplicate_dates:
        warnings.append(
            "Обнаружены повторяющиеся даты прогноза; анализ остановлен, "
            "так как для одной даты задано несколько остатков: "
            + ", ".join(day.isoformat() for day in duplicate_dates[:10])
            + ("…" if len(duplicate_dates) > 10 else "")
        )
        return {
            **empty_result,
            "warnings": warnings,
        }

    gaps = []
    for previous, current in zip(dates, dates[1:]):
        delta = (current - previous).days
        if delta > 1:
            gaps.append((previous + timedelta(days=1), current - timedelta(days=1)))

    if gaps:
        missing_days = sum((end - start).days + 1 for start, end in gaps)
        preview = ", ".join(
            start.isoformat() if start == end
            else f"{start.isoformat()}—{end.isoformat()}"
            for start, end in gaps[:4]
        )
        warnings.append(
            f"Прогноз неполный: отсутствуют {missing_days} календарных дн. "
            f"({preview}{'…' if len(gaps) > 4 else ''}). "
            "Пропущенные даты не считались нулевым остатком; события "
            "по разные стороны пропуска разделены."
        )

    current_balance: float | None = None
    if current_col is not None:
        values = []
        for value in frame[current_col]:
            if not _is_missing(value):
                try:
                    number = float(value)
                    if math.isfinite(number):
                        values.append(number)
                except (ValueError, TypeError):
                    pass
        if values:
            current_balance = values[0]
    else:
        warnings.append(
            "Текущий остаток не передан: уровень риска определён по "
            "длительности; оценка относительного размера дефицита недоступна."
        )

    minimum_balance = min(item["balance"] for item in normalized)
    maximum_deficit = max(0.0, -minimum_balance)

    events_raw: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []

    for item in normalized:
        if item["balance"] < 0:
            if active and (item["date"] - active[-1]["date"]).days == 1:
                active.append(item)
            else:
                if active:
                    events_raw.append(active)
                active = [item]
        else:
            if active:
                events_raw.append(active)
                active = []

    if active:
        events_raw.append(active)

    events: list[dict[str, Any]] = []
    for index, run in enumerate(events_raw, start=1):
        start = run[0]["date"]
        end = run[-1]["date"]
        event_minimum = min(item["balance"] for item in run)
        deficit = abs(event_minimum)
        duration = (end - start).days + 1
        level = _risk_level(
            deficit, duration, current_balance, thresholds
        )
        events.append(
            {
                "risk_id": f"cash-gap-{index:03d}",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "minimum_balance": round(event_minimum, 2),
                "maximum_deficit": round(deficit, 2),
                "duration_days": duration,
                "risk_level": level,
                "explanation": _explanation(
                    start, end, event_minimum, deficit, duration,
                    current_balance, level
                ),
            }
        )

    if not events:
        status = "no_gap"
    elif any(event["risk_level"] == "high" for event in events):
        status = "high"
    elif any(event["risk_level"] == "medium" for event in events):
        status = "medium"
    else:
        status = "low"

    if (gaps or invalid_rows) and not events:
        status = "incomplete"
    elif (gaps or invalid_rows) and events:
        warnings.append(
            "Риски рассчитаны только для дат с валидным прогнозом; "
            "пропуски могут скрывать дополнительные кассовые разрывы."
        )

    return {
        "events": events,
        "status": status,
        "minimum_balance": round(minimum_balance, 2),
        "maximum_deficit": round(maximum_deficit, 2),
        "warnings": warnings,
        "probability": None,
        "probability_status": "not_estimated",
    }


def analyze_cash_gap(
    forecast: Any,
    *,
    thresholds: RiskThresholds | None = None,
) -> dict[str, Any]:
    """Return events, overall risk status, extrema, warnings and probability.

    `probability` is always None in this rule-based module. It must only be
    populated by a separately trained and validated probabilistic model.
    """
    return _analyze(forecast, thresholds=thresholds)
