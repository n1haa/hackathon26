"""Adapter for externally detected recurring occurrences; no recurrence detection."""
import pandas as pd

from .data import DIRECTIONS, day, money, _present


def separate_events(prepared, recurring_events, horizon, timezone):
    """Remove matched history before fitting, add only explicit dated occurrences.

    Every historical recurring_id must have an event (amount zero can cancel it).
    A new obligation must explicitly set is_new=True and must not match history.
    Matching by text/category/amount is intentionally unsupported.
    """
    daily = prepared.daily.copy()
    if recurring_events is None:
        return daily, [], []
    events = pd.DataFrame(recurring_events).to_dict("records")
    if not events:
        return daily, [], []
    end = prepared.as_of + pd.Timedelta(days=horizon)
    seen = set()
    occurrences = set()
    series = set()
    normalized = []
    history = prepared.transactions
    history_ids = (set(history["recurring_id"].dropna().astype(str))
                   if "recurring_id" in history else set())
    for event in events:
        if not all(k in event for k in ("event_id", "recurring_id", "date", "amount", "direction")):
            raise ValueError("Events require event_id, recurring_id, date, amount, direction")
        for key in ("event_id", "recurring_id"):
            if pd.isna(event[key]) or not str(event[key]).strip():
                raise ValueError(f"Event {key} cannot be empty")
        event_id, recurring_id = str(event["event_id"]), str(event["recurring_id"])
        if event_id in seen:
            raise ValueError("Duplicate event_id; each occurrence must be unique")
        seen.add(event_id)
        date = day(event["date"], timezone)
        if date <= prepared.as_of or date > end:
            raise ValueError("Recurring event date must be inside the forecast horizon")
        direction = DIRECTIONS.get(str(event["direction"]).strip().lower())
        if direction is None:
            raise ValueError("Invalid recurring event direction")
        amount = money(event["amount"], "event amount", nonnegative=True)
        if event.get("category") == "internal_transfer":
            raise ValueError("Internal transfers must not be supplied as external recurring events")
        is_new = event.get("is_new", False)
        if not _present(is_new):
            is_new = False
        if not isinstance(is_new, bool):
            raise ValueError("is_new must be boolean")
        if is_new == (recurring_id in history_ids):
            raise ValueError("Recurring events must match historical recurring_id, or explicitly set is_new=True for a new series")
        occurrence = (recurring_id, date, direction)
        if occurrence in occurrences:
            raise ValueError("Duplicate recurring occurrence even with different event_id")
        occurrences.add(occurrence)
        if recurring_id in history_ids:
            historical_directions = set(history.loc[history["recurring_id"].astype(str).eq(recurring_id), "direction"])
            if historical_directions != {direction}:
                raise ValueError("Recurring series direction must match its historical operations")
        series.add(recurring_id)
        normalized.append({"event_id": event_id, "recurring_id": recurring_id,
                           "date": date.date().isoformat(), "amount": amount,
                           "direction": direction, "is_new": is_new})
    # A partial schedule would erase historical events whose future is unspecified.
    if not history_ids.issubset(series):
        raise ValueError("Supply the complete horizon schedule for all historical recurring_id values; use a zero-amount cancellation if needed")
    matched = history[history["recurring_id"].astype(str).isin(series)] if "recurring_id" in history else history.iloc[:0]
    if matched["category"].eq("internal_transfer").any():
        raise ValueError("Recurring series cannot contain internal transfers")
    for row in matched.to_dict("records"):
        target = "income" if row["direction"] == "incoming" else "expense"
        date = row["date"]
        daily.loc[date, target] -= float(row["amount"])
        column = f"category::{row['direction']}::{row['category']}"
        if column in daily:
            daily.loc[date, column] -= float(row["amount"])
    # Decimal input has exact cents; clean tiny float subtraction artifacts.
    for column in [c for c in daily if c in ("income", "expense") or c.startswith("category::")]:
        daily[column] = daily[column].round(2).clip(lower=0)
    daily["net_flow"] = daily["income"] - daily["expense"]
    note = ["Recurring history removed by recurring_id; only explicit scheduled occurrences added. "
            "Metrics evaluate residual historical flows, not future schedule accuracy."]
    return daily, normalized, note
