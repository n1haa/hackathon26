"""
CashFlow AI: recurring financial operation detector.

Input schema follows README.md:
company_id, operation_id, operation_date, amount, direction,
counterparty_inn, counterparty_name, payment_purpose, balance_eod.

`category` is optional because it is not present in the supplied dataset.
The detector preserves it when an upstream module provides it. If absent,
it does not invent a category; normalized payment-purpose text is used as
a conservative grouping feature.

Confidence is a heuristic score for regularity detection, NOT the probability
that a future payment will occur.
"""
from __future__ import annotations

import calendar
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATE_COLUMNS = ("operation_date", "date", "transaction_date")
AMOUNT_COLUMNS = ("amount", "operation_amount", "sum")
DIRECTION_COLUMNS = ("direction", "operation_type", "type")
INN_COLUMNS = ("counterparty_inn", "inn")
NAME_COLUMNS = ("counterparty_name", "counterparty")
PURPOSE_COLUMNS = ("payment_purpose", "purpose", "description")
CATEGORY_COLUMNS = ("category", "operation_category")

_MONTHS_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
_STOPWORDS = {
    "оплата", "оплате", "оплаты", "платеж", "платежа", "платежу",
    "перечисление", "перевод", "по", "за", "от", "для", "согласно",
    "договор", "договору", "счет", "счёт", "счета", "счету",
    "упд", "акт", "накладная", "накладной", "ндс", "без", "не",
    "облагается", "руб", "рублей", "р", "номер", "№",
}
_NOISE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), " "),
    (re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"), " "),
    (re.compile(r"\b(?:№\s*)?\d{2,}\b"), " "),
    (re.compile(r"\b(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b"), " месяц "),
    (re.compile(r"\b(?:январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)\b"), " месяц "),
    (re.compile(r"\b(?:без\s+ндс|ндс\s+не\s+облагается)\b"), " "),
]


def _first_existing(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    if required:
        raise ValueError(f"Не найден обязательный столбец. Ожидался один из: {', '.join(candidates)}")
    return None


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text in {"nan", "none", "null", "n/a", "-"} else text


def _normalize_purpose(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    # Keep meaningful words, but remove common document and accounting boilerplate.
    tokens = re.findall(r"[a-zа-я]+|[a-zа-я]+-[a-zа-я]+", text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens[:12])


def _canonical_direction(value: Any) -> str:
    text = _clean(value)
    if text in {"поступление", "income", "inflow", "credit", "приход"}:
        return "income"
    if text in {"списание", "expense", "outflow", "debit", "расход"}:
        return "expense"
    # Do not guess unknown directions: preserve the normalized source value.
    return text or "unknown"


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except (TypeError, ValueError, OverflowError):
        return None


def _month_add(year: int, month: int, offset: int, day: int) -> date:
    idx = year * 12 + (month - 1) + offset
    y, m0 = divmod(idx, 12)
    m = m0 + 1
    return date(y, m, min(day, calendar.monthrange(y, m)[1]))


def _previous_weekday(d: date) -> date:
    # The supplied statements contain working-day dates. We only adjust weekends;
    # Russian public holidays cannot be inferred from this dataset alone.
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _expected_next_date(dates: list[date], frequency: str, interval_days: float) -> date:
    last = max(dates)
    if frequency == "monthly":
        # Estimate a recurring day-of-month from observed operations, then find
        # the next calendar month. Clamp month-end (e.g. 31st -> 28/29/30).
        dom = int(round(float(np.median([d.day for d in dates]))))
        candidate = _month_add(last.year, last.month, 1, dom)
        if candidate <= last:
            candidate = _month_add(last.year, last.month, 2, dom)
        return _previous_weekday(candidate)
    if frequency == "biweekly":
        step = 14
    elif frequency == "weekly":
        step = 7
    elif frequency == "daily":
        step = 1
    else:
        step = max(1, int(round(interval_days)))
    candidate = last + timedelta(days=step)
    adjusted = _previous_weekday(candidate)
    if adjusted <= last:
        # A daily payment after Friday must not be predicted on that same Friday.
        adjusted = candidate
        while adjusted.weekday() >= 5:
            adjusted += timedelta(days=1)
    return adjusted


def _frequency_label(interval: float) -> str:
    if 0.5 <= interval <= 1.8:
        return "daily"
    if 5.0 <= interval <= 9.5:
        return "weekly"
    if 11.0 <= interval <= 17.5:
        return "biweekly"
    if 25.0 <= interval <= 35.5:
        return "monthly"
    return "every_n_days"


def _interval_consistency(intervals: np.ndarray, expected: float) -> float:
    if len(intervals) == 0 or expected <= 0:
        return 0.0
    # Relative median absolute deviation is robust to occasional delays.
    med = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - med)))
    cv_robust = 1.4826 * mad / max(med, 1.0)
    center_match = math.exp(-abs(med - expected) / max(expected * 0.35, 1.0))
    dispersion_match = math.exp(-cv_robust / 0.35)
    return float(np.clip(0.65 * center_match + 0.35 * dispersion_match, 0, 1))


def _amount_stability(amounts: np.ndarray) -> float:
    if len(amounts) < 2:
        return 0.0
    med = float(np.median(np.abs(amounts)))
    if med <= 0:
        return 0.0
    mad = float(np.median(np.abs(amounts - np.median(amounts))))
    robust_cv = 1.4826 * mad / med
    return float(math.exp(-robust_cv / 0.5))


def _calendar_match(dates: list[date], frequency: str) -> float:
    if frequency == "monthly":
        # Monthly transactions may move around month-end / weekends. Compare
        # day-of-month dispersion, allowing month-end clamping.
        dom = np.asarray([d.day for d in dates], dtype=float)
        med = float(np.median(dom))
        mad = float(np.median(np.abs(dom - med)))
        return float(math.exp(-mad / 5.0))
    if frequency in {"weekly", "biweekly"}:
        weekdays = [d.weekday() for d in dates]
        counts = Counter(weekdays)
        return max(counts.values()) / len(weekdays)
    if frequency == "daily":
        return float(np.mean(np.asarray([d.weekday() for d in dates]) < 5))
    return 0.5


def _candidate_group_key(row: pd.Series, cols: dict[str, str | None]) -> tuple:
    inn = _clean(row[cols["inn"]]) if cols["inn"] else ""
    name = _clean(row[cols["name"]]) if cols["name"] else ""
    purpose = _normalize_purpose(row[cols["purpose"]]) if cols["purpose"] else ""
    category = _clean(row[cols["category"]]) if cols["category"] else ""
    direction = _canonical_direction(row[cols["direction"]])

    # Prefer stable counterparty identity; do not use amount as a grouping key.
    if inn:
        identity = ("inn", inn)
    elif name and name not in {"физическое лицо", "не указан", "unknown"}:
        identity = ("name", name)
    elif purpose:
        # Without counterparty identity, only group when the normalized purpose
        # fingerprint is available; similar-looking empty-purpose operations stay out.
        identity = ("purpose", purpose)
    else:
        return ("unmatched", str(row.get("__row_id", "")))

    # If no upstream category exists, the normalized purpose becomes a cautious
    # sub-grouping feature to avoid merging unrelated transactions for one bank.
    text_key = purpose if purpose else "__empty_purpose__"
    return (identity, direction, category, text_key)


def detect_recurring(transactions: pd.DataFrame | list[dict[str, Any]], *,
                     min_occurrences: int = 3,
                     min_confidence: float = 0.55) -> list[dict[str, Any]]:
    """
    Detect recurring incoming/outgoing operations.

    Parameters
    ----------
    transactions:
        DataFrame or list of dictionaries. Dataset column names from README.md
        are supported. `category` is optional and preserved if present.
    min_occurrences:
        Minimum number of distinct operation dates in a group (default 3).
    min_confidence:
        Minimum heuristic regularity score in [0, 1] for inclusion.

    Returns
    -------
    list[dict]:
        Events with the requested output fields. `confidence` is a heuristic
        regularity score, not a probability of a future payment.
    """
    if min_occurrences < 3:
        raise ValueError("min_occurrences должен быть не меньше 3")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence должен быть в диапазоне [0, 1]")

    df = transactions.copy() if isinstance(transactions, pd.DataFrame) else pd.DataFrame(transactions)
    if df.empty:
        return []

    # Case-insensitive matching while preserving actual source column labels.
    date_col = _first_existing(df.columns, DATE_COLUMNS)
    amount_col = _first_existing(df.columns, AMOUNT_COLUMNS)
    direction_col = _first_existing(df.columns, DIRECTION_COLUMNS)
    cols = {
        "direction": direction_col,
        "inn": _first_existing(df.columns, INN_COLUMNS, required=False),
        "name": _first_existing(df.columns, NAME_COLUMNS, required=False),
        "purpose": _first_existing(df.columns, PURPOSE_COLUMNS, required=False),
        "category": _first_existing(df.columns, CATEGORY_COLUMNS, required=False),
    }

    work = df.copy()
    work["__row_id"] = np.arange(len(work))
    parsed_dates = pd.to_datetime(work[date_col], errors="coerce")
    work["__date"] = parsed_dates.dt.date
    work["__amount"] = pd.to_numeric(work[amount_col], errors="coerce")
    work["__direction"] = work[direction_col].map(_canonical_direction)
    work = work[work["__date"].notna() & np.isfinite(work["__amount"]) & (work["__amount"] > 0)].copy()
    if work.empty:
        return []

    # Analyze each company independently when company_id exists; transactions
    # from different companies must never be combined.
    company_col = _first_existing(work.columns, ("company_id",), required=False)
    # Avoid DataFrame.apply(axis=1) on the 222k-row dataset; pre-normalize columns
    # and build keys from arrays for a materially faster pass.
    inn_values = work[cols["inn"]].map(_clean) if cols["inn"] else pd.Series("", index=work.index)
    name_values = work[cols["name"]].map(_clean) if cols["name"] else pd.Series("", index=work.index)
    purpose_values = work[cols["purpose"]].map(_normalize_purpose) if cols["purpose"] else pd.Series("", index=work.index)
    category_values = work[cols["category"]].map(_clean) if cols["category"] else pd.Series("", index=work.index)
    direction_values = work[direction_col].map(_canonical_direction)
    group_keys = []
    for row_id, inn, name, purpose, category, direction in zip(
        work["__row_id"].to_numpy(),
        inn_values.to_numpy(),
        name_values.to_numpy(),
        purpose_values.to_numpy(),
        category_values.to_numpy(),
        direction_values.to_numpy(),
    ):
        if inn:
            identity = ("inn", inn)
        elif name and name not in {"физическое лицо", "не указан", "unknown"}:
            identity = ("name", name)
        elif purpose:
            identity = ("purpose", purpose)
        else:
            identity = ("unmatched", str(row_id))
        # Purpose is used as a cautious sub-grouping key when the supplied data
        # has no category column; amount is never used to group operations.
        text_key = purpose if purpose else "__empty_purpose__"
        group_keys.append((identity, direction, category, text_key))
    work["__group_key"] = group_keys
    if company_col:
        work["__company_key"] = work[company_col].map(_clean)
    else:
        work["__company_key"] = ""

    results: list[dict[str, Any]] = []
    group_cols = ["__company_key", "__group_key"]
    for (_, group_key), group in work.groupby(group_cols, sort=False, dropna=False):
        # Distinct operations on the same date should not artificially increase
        # recurrence count. Aggregate same-day amounts before interval analysis.
        group = group.sort_values("__date")
        daily = group.groupby("__date", as_index=False).agg(
            amount=("__amount", "sum"),
            direction=("__direction", "first"),
        ).sort_values("__date")
        dates = list(daily["__date"])
        amounts = daily["amount"].to_numpy(dtype=float)
        if len(dates) < min_occurrences:
            continue

        intervals = np.asarray([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))], dtype=float)
        if len(intervals) < 2 or np.any(intervals <= 0):
            continue
        median_interval = float(np.median(intervals))
        frequency = _frequency_label(median_interval)

        # Estimate missing cycles: an interval near 2x/3x the typical interval
        # counts as a missed expected occurrence, not a new frequency.
        expected_interval = median_interval
        if frequency == "daily":
            expected_interval = 1.0
        elif frequency == "weekly":
            expected_interval = 7.0
        elif frequency == "biweekly":
            expected_interval = 14.0
        elif frequency == "monthly":
            # Calendar monthly intervals are variable; 30 days is only for
            # confidence calculation, while expected date uses calendar months.
            expected_interval = 30.0

        interval_score = _interval_consistency(intervals, expected_interval)
        amount_score = _amount_stability(amounts)
        calendar_score = _calendar_match(dates, frequency)

        # Penalize gaps that look like skipped cycles, but tolerate small delays.
        ratios = intervals / max(expected_interval, 1.0)
        missed = np.maximum(0, np.rint(ratios) - 1)
        gap_penalty = float(np.mean(np.exp(-0.65 * missed)))
        count_score = min(1.0, (len(dates) - 2) / 4.0 + 0.5)

        confidence = float(np.clip(
            0.30 * count_score
            + 0.30 * interval_score
            + 0.20 * amount_score
            + 0.10 * gap_penalty
            + 0.10 * calendar_score,
            0.0, 1.0
        ))
        if confidence < min_confidence:
            continue

        # Recover source labels from the first row in the group.
        first = group.iloc[0]
        counterparty = None
        if cols["name"]:
            val = _clean(first[cols["name"]])
            counterparty = str(first[cols["name"]]).strip() if val else None
        if not counterparty and cols["inn"]:
            val = _clean(first[cols["inn"]])
            counterparty = str(first[cols["inn"]]).strip() if val else None

        category = None
        if cols["category"]:
            val = _clean(first[cols["category"]])
            category = str(first[cols["category"]]).strip() if val else None

        direction = _canonical_direction(first[direction_col])
        direction = {"income": "Поступление", "expense": "Списание"}.get(direction, direction)
        expected_date = _expected_next_date(dates, frequency, median_interval)

        company_id = None
        if company_col:
            company_value = first[company_col]
            company_id = str(company_value).strip() if _clean(company_value) else None

        results.append({
            "company_id": company_id,
            "counterparty": counterparty,
            "category": category,
            "direction": direction,
            "frequency": frequency,
            "median_amount": round(float(np.median(amounts)), 2),
            "expected_amount": round(float(np.median(amounts)), 2),
            "expected_next_date": expected_date.isoformat(),
            "interval_days": round(median_interval, 2),
            "occurrences_count": int(len(dates)),
            "confidence": round(confidence, 3),
            "source": "history",
        })

    results.sort(key=lambda event: (-event["confidence"], event["expected_next_date"], str(event["counterparty"] or "")))
    return results
