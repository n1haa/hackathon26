"""Reproducible SYNTHETIC data, not a bank dataset or evidence of real accuracy."""
from decimal import Decimal

import numpy as np
import pandas as pd

from .evaluation import positive_integer


def make_synthetic_transactions(days=240, seed=42):
    """A single fictional RUB business, weekly activity, trend, noise and rent.

    Returns transactions and complete-coverage/balance metadata. No generated
    future obligations are passed to the forecasting API.
    """
    days = positive_integer(days, "days")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    opening = Decimal("100000.00")
    balance = opening
    records = []
    for i, date in enumerate(dates):
        weekend = date.dayofweek >= 5
        income = max(0., (5200 if weekend else 11000) + 12 * i + rng.normal(0, 550))
        expense = max(0., (1700 if weekend else 4300) + 5 * i + rng.normal(0, 300))
        # Confirmed no-operation dates test coverage; preserve them in metadata.
        if i % 37 == 17:
            continue
        flows = [("incoming", income, "revenue", "Оплата за услуги"),
                 ("outgoing", expense, "purchases", "Закупка сырья")]
        if date.day == 5:
            flows.append(("outgoing", 25000, "rent", "Аренда помещения"))
        for j, (direction, amount, category, purpose) in enumerate(flows):
            value = Decimal(str(round(amount, 2)))
            balance += value if direction == "incoming" else -value
            records.append({"transaction_id": f"synthetic-{i}-{j}",
                            "date": (date + pd.Timedelta(hours=10 + j)).isoformat(),
                            "direction": direction, "amount": float(value),
                            "category": category, "purpose": purpose,
                            "closing_balance": float(balance), "currency": "RUB",
                            "company_id": "SYNTHETIC-DEMO", "synthetic": True})
    return records, {"synthetic": True, "seed": seed, "days": days,
                     "history_start": dates[0].date().isoformat(),
                     "as_of": dates[-1].date().isoformat(), "complete_history": True,
                     "opening_balance": float(opening), "current_balance": float(balance)}
