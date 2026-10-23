"""Statement normalization, explicit coverage and balance reconciliation."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import numpy as np
import pandas as pd


CENT = Decimal("0.01")
DIRECTIONS = {"incoming": "incoming", "outgoing": "outgoing",
              "поступление": "incoming", "списание": "outgoing"}


def rounded_prediction(value):
    """One shared rounding rule for evaluation and the delivered forecast."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def money(value, name="amount", nonnegative=False):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError()
        result = Decimal(str(value))
        if not result.is_finite() or (nonnegative and result < 0):
            raise ValueError()
        if result != result.quantize(CENT):
            raise ValueError()
        return result
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite money with at most two decimals"
                         + (" and nonnegative" if nonnegative else "")) from exc


def day(value, timezone="Europe/Moscow"):
    try:
        if value is None or isinstance(value, (bool, int, float, np.number)):
            raise ValueError()
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError()
        if result.tzinfo is not None:
            result = result.tz_convert(timezone).tz_localize(None)
        return result.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date: {value!r}") from exc


def _timestamp(value, timezone):
    day(value, timezone)
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize(timezone)
    return result.tz_convert(timezone)


def _present(value):
    return value is not None and not pd.isna(value)


@dataclass
class PreparedCashflows:
    daily: pd.DataFrame
    transactions: pd.DataFrame
    warnings: list[str]
    as_of: pd.Timestamp


def prepare_daily_cashflows(transactions, *, as_of=None, history_start=None,
                           complete_history=False, observed_dates=None,
                           opening_balance=None, current_balance=None,
                           timezone="Europe/Moscow", fields=None):
    """Unknown dates retain NaN; only explicit statement coverage permits zero.

    closing_balance is after the transaction, per account when account_id exists,
    otherwise consolidated. Input order breaks ties between identical timestamps.
    opening_balance is before history_start and supported for a single account.
    """
    frame = pd.DataFrame(transactions).copy()
    if fields:
        frame = frame.rename(columns={source: target for target, source in fields.items()})
    required = {"date", "amount", "direction"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Nonempty transactions with date, amount, direction are required")
    for name in ("company_id", "currency"):
        if name in frame and (frame[name].isna().any() or frame[name].nunique() != 1):
            raise ValueError(f"Supply exactly one nonmissing {name}; consolidate/convert upstream")
    if "transaction_id" in frame:
        ids = frame["transaction_id"].dropna()
        if ids.duplicated().any():
            raise ValueError("Duplicate transaction_id; deduplicate upstream")
    frame["timestamp"] = [_timestamp(v, timezone) for v in frame["date"]]
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame["date"] = frame["timestamp"].dt.tz_localize(None).dt.normalize()
    amounts = [money(v, nonnegative=True) for v in frame["amount"]]
    frame["amount"] = amounts
    frame["direction"] = frame["direction"].map(lambda v: DIRECTIONS.get(str(v).lower().strip()))
    if frame["direction"].isna().any():
        raise ValueError("direction must be incoming/outgoing or поступление/списание")
    frame["category"] = frame.get("category", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    frame["signed_amount"] = [a if d == "incoming" else -a
                               for a, d in zip(amounts, frame["direction"])]
    internal = frame["category"].eq("internal_transfer")
    frame["external_amount"] = [Decimal(0) if flag else amount
                                for flag, amount in zip(internal, amounts)]
    end = day(as_of, timezone) if as_of is not None else frame["date"].max()
    start = day(history_start, timezone) if history_start is not None else frame["date"].min()
    if start > end or frame["date"].min() < start or frame["date"].max() > end:
        raise ValueError("Transactions outside history_start/as_of; future data are forbidden")
    index = pd.date_range(start, end, freq="D", name="date")
    if observed_dates is not None and complete_history:
        raise ValueError("Use complete_history OR observed_dates")
    if observed_dates is None:
        observed = index if complete_history else pd.DatetimeIndex(frame["date"].unique())
    else:
        observed = pd.DatetimeIndex([day(v, timezone) for v in observed_dates])
        if not observed.isin(index).all() or not frame["date"].isin(observed).all():
            raise ValueError("observed_dates must cover all transaction dates within history")
    daily = pd.DataFrame(index=index)
    daily["observed"] = index.isin(observed)
    for direction, target in (("incoming", "income"), ("outgoing", "expense")):
        rows = frame[frame["direction"].eq(direction)]
        totals = rows.groupby("date")["external_amount"].sum()
        daily[target] = totals.reindex(index, fill_value=Decimal(0)).astype(float)
        for category in sorted(rows.loc[rows["category"].ne("internal_transfer"), "category"].unique()):
            values = rows[rows["category"].eq(category)].groupby("date")["external_amount"].sum()
            daily[f"category::{direction}::{category}"] = values.reindex(index, fill_value=Decimal(0)).astype(float)
    flow_columns = [c for c in daily if c != "observed"]
    daily.loc[~daily["observed"], flow_columns] = np.nan
    daily["net_flow"] = daily["income"] - daily["expense"]
    messages = []
    if not daily["observed"].all():
        messages.append("Unknown statement coverage: missing dates remain NaN, not zero. "
                        "Provide complete_history=True only for a complete statement or observed_dates.")
    _reconcile(frame, opening_balance, current_balance, messages)
    return PreparedCashflows(daily, frame, messages, end)


def _reconcile(frame, opening_balance, current_balance, messages):
    account_specific = "account_id" in frame
    if account_specific and frame["account_id"].isna().any():
        raise ValueError("account_id cannot be missing when supplied")
    groups = list(frame.groupby("account_id", sort=False)) if account_specific else [("consolidated", frame)]
    if opening_balance is not None and len(groups) != 1:
        raise ValueError("opening_balance requires a single account; use per-account closing snapshots")
    final_balances = []
    for account, rows in groups:
        expected = money(opening_balance, "opening_balance") if opening_balance is not None else None
        for row in rows.to_dict("records"):
            change = row["signed_amount"]
            if not account_specific and row["category"] == "internal_transfer":
                change = Decimal(0)
            if expected is not None:
                expected += change
            if _present(row.get("closing_balance")):
                actual = money(row["closing_balance"], "closing_balance")
                if expected is not None and actual != expected:
                    messages.append(f"Balance mismatch for {account} at {row['timestamp']}: "
                                    f"statement={actual}, operations={expected}, difference={actual - expected}. "
                                    "Forecast uses supplied current_balance; missing or incorrect flows can bias it.")
                expected = actual
        final_balances.append(expected)
    if current_balance is not None:
        current = money(current_balance, "current_balance")
        if all(v is not None for v in final_balances):
            expected = sum(final_balances, Decimal(0))
            if expected != current:
                messages.append(f"current_balance mismatch: supplied={current}, statement={expected}. "
                                "Forecast is anchored to supplied current_balance; the discrepancy shifts every forecast balance.")
    if opening_balance is None:
        messages.append("No opening_balance: movements before each account's first closing snapshot cannot be reconciled.")
    if any(v is None for v in final_balances):
        messages.append("Insufficient balance snapshots: current_balance cannot be fully reconciled.")
