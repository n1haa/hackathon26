"""CSV loading and validation at the boundary of tasks 2 and 4.

This does not categorize operations or reconstruct bank balances. Invalid
required values and duplicate operation IDs are rejected, never silently lost.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "operation_date": ("date", "transaction_date"),
    "amount": ("operation_amount", "sum"),
    "direction": ("operation_type", "type"),
    "counterparty_inn": ("inn",),
    "counterparty_name": ("counterparty",),
    "payment_purpose": ("purpose", "description"),
    "category": ("operation_category",),
    "balance_eod": ("balance",),
}
DIRECTIONS = {
    **dict.fromkeys(("поступление", "income", "inflow", "credit", "приход"), "Поступление"),
    **dict.fromkeys(("списание", "expense", "outflow", "debit", "расход"), "Списание"),
}


def validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy. Optional category and missing INNs are preserved."""
    frame = transactions.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Повторные названия столбцов операций")
    for target, aliases in ALIASES.items():
        present = [name for name in (target, *aliases) if name in frame]
        if len(present) > 1:
            raise ValueError(f"Неоднозначные столбцы {target}: {present}")
        if present:
            frame.rename(columns={present[0]: target}, inplace=True)
    if frame.empty:
        return frame
    missing = {"operation_date", "amount", "direction"} - set(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные поля: {sorted(missing)}")
    dates = frame.operation_date.map(
        lambda value: pd.NaT if isinstance(value, (int, float, bool))
        else pd.to_datetime(value, errors="coerce")
    )
    if dates.isna().any():
        raise ValueError("Некорректная operation_date")
    frame["operation_date"] = dates.map(lambda value: value.date().isoformat())
    amounts = pd.to_numeric(frame.amount, errors="coerce")
    if amounts.isna().any() or not np.isfinite(amounts).all() or amounts.le(0).any():
        raise ValueError("amount должна быть конечной положительной суммой; знак задаёт direction")
    frame["amount"] = amounts
    directions = frame.direction.astype("string").str.strip().str.lower().map(DIRECTIONS)
    if directions.isna().any():
        raise ValueError("Неизвестное или пустое direction")
    frame["direction"] = directions
    for column in ("company_id", "operation_id", "counterparty_inn"):
        if column in frame:
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value).strip())
    if "company_id" in frame and (frame.company_id.isna() | frame.company_id.eq("")).any():
        raise ValueError("company_id не должен быть пустым, если столбец передан")
    if "operation_id" in frame:
        if (frame.operation_id.isna() | frame.operation_id.eq("")).any():
            raise ValueError("operation_id не должен быть пустым, если столбец передан")
        key = ["company_id", "operation_id"] if "company_id" in frame else ["operation_id"]
        if frame.duplicated(key).any():
            raise ValueError("Повторный operation_id: устраните дубликаты перед прогнозом")
    if "balance_eod" in frame:
        balances = pd.to_numeric(frame.balance_eod, errors="coerce")
        if balances.isna().any() or not np.isfinite(balances).all():
            raise ValueError("Некорректный balance_eod")
        frame["balance_eod"] = balances
    return frame.sort_values("operation_date", kind="stable").reset_index(drop=True)


def load_transactions(path: str | Path, *, company_id: str | None = None) -> pd.DataFrame:
    """Read the supplied semicolon CSV, retaining leading zeros in identifiers.

    Multi-company files require explicit company selection. XLSX loading belongs
    to the upstream ingestion module and is outside this integration adapter.
    """
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame.columns = frame.columns.str.strip().str.lower()
    if company_id is not None:
        if "company_id" not in frame:
            raise ValueError("Для выбора компании требуется company_id")
        frame = frame[frame.company_id.str.strip().eq(str(company_id).strip())].copy()
        if frame.empty:
            raise ValueError(f"Компания {company_id!r} не найдена")
    if "company_id" in frame and frame.company_id.nunique() > 1:
        raise ValueError("CSV содержит несколько компаний: укажите company_id")
    return validate_transactions(frame)
