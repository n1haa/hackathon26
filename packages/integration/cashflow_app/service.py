"""Translate boundary schemas without changing the original financial algorithms."""
from io import BytesIO
from pathlib import Path
from decimal import Decimal
import json

import pandas as pd

from cashflow_integrated import run_cashflow_pipeline
from cashflow_integrated.data_loader import validate_transactions
from forecasting import forecast_cashflow
from forecasting.data import money
from transaction_classifier import classify_transactions


def read_statement(source, *, filename=None):
    """Read identifiers as strings, including leading zeroes; never deserialize models."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename, contents = path.name, path.read_bytes()
    else:
        contents = source if isinstance(source, bytes) else source.getvalue()
    if len(contents) > 20 * 1024 * 1024:
        raise ValueError("Размер файла превышает 20 МБ; выгрузите выписку одной компании")
    suffix = Path(filename or "statement.csv").suffix.lower()
    if suffix == ".xlsx":
        frame = pd.read_excel(BytesIO(contents), dtype=str)
    elif suffix == ".csv":
        try:
            text = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = contents.decode("cp1251")
        from io import StringIO
        frame = pd.read_csv(StringIO(text), sep=None, engine="python", dtype=str)
    else:
        raise ValueError("Поддерживаются только CSV и XLSX")
    frame.columns = frame.columns.str.strip().str.lower()
    if frame.empty:
        raise ValueError("Выписка не содержит операций")
    if frame.columns.duplicated().any():
        raise ValueError("Повторные названия столбцов")
    return frame


def analyze_statement(frame, *, current_balance, complete_history=False,
                      company_id=None, horizon=30, own_inn=None, model="auto"):
    """One company, currency and account, at the last statement date.

    Recurrence signals are reported separately: the detector does not supply the
    historical membership and complete schedule needed to remove modeled flows.
    """
    frame = pd.DataFrame(frame).copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Повторные названия столбцов")
    if company_id is not None:
        if "company_id" not in frame:
            raise ValueError("Для выбора компании требуется company_id")
        frame = frame[frame.company_id.astype("string").str.strip().eq(str(company_id).strip())].copy()
    if frame.empty:
        raise ValueError("Нет операций выбранной компании")
    for column in ("company_id", "account_id", "currency"):
        if column in frame:
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or values.eq("").any() or values.nunique() != 1:
                raise ValueError(f"Требуется одно непустое значение {column}; разделите выписку")
            frame[column] = values
    # Upstream risk loader uses Russian directions; forecasting also accepts them.
    if "direction" in frame:
        frame["direction"] = frame.direction.replace({"incoming": "Поступление", "outgoing": "Списание"})
    # Reject booleans / fractions before pandas coerces them into plausible money.
    for column in ("amount", "operation_amount", "sum", "balance_eod", "balance"):
        if column in frame:
            for value in frame[column]:
                money(value, column, nonnegative=column in {"amount", "operation_amount", "sum"})
    tx = validate_transactions(frame)
    balance = money(current_balance, "current_balance")
    if not isinstance(complete_history, bool):
        raise ValueError("complete_history должен быть boolean")
    warnings = []
    if "balance_eod" in tx:
        # Repeated EOD values are snapshots, never per-operation balances or sums.
        daily_balances = tx.groupby("operation_date").balance_eod
        if daily_balances.nunique().gt(1).any():
            raise ValueError("Противоречивые balance_eod в один день")
        last_balance = money(tx.balance_eod.iloc[-1])
        if balance != last_balance:
            raise ValueError("Текущий остаток должен совпадать с balance_eod последнего дня выписки")
        previous = None
        for day, rows in tx.groupby("operation_date", sort=True):
            closing = money(rows.balance_eod.iloc[0])
            flow = sum((money(row.amount) * (1 if row.direction == "Поступление" else -1)
                        for row in rows.itertuples()), Decimal(0))
            if previous is not None and closing != previous + flow:
                warnings.append(f"Остаток за {day} не сходится с потоком операций; проверьте полноту выписки.")
            previous = closing
    tx = classify_transactions(tx, fields={"date": "operation_date", "purpose": "payment_purpose"},
                               own_inns={str(own_inn).strip()} if own_inn else set())
    forecast = forecast_cashflow(
        tx, current_balance=balance, horizon=horizon,
        fields={"date": "operation_date", "transaction_id": "operation_id"},
        complete_history=complete_history, model=model,
    )
    # The risk engine needs the starting balance on each row for relative severity.
    risk_rows = [{**row, "current_balance": float(balance)} for row in forecast["forecast"]]
    risk = run_cashflow_pipeline(tx, forecast=risk_rows, horizon_days=horizon)
    warnings.extend(forecast["warnings"])
    warnings.extend(risk["warnings"])
    warnings.append("Регулярные события показаны отдельно: прогноз учитывает исторические потоки, "
                    "но не добавляет неполное расписание повторно. Вероятность кассового разрыва не оценивалась.")
    if not own_inn:
        warnings.append("ИНН компании не задан: принадлежность переводов между своими счетами не подтверждается.")
    return {"company_id": risk["company_id"], "transactions": json.loads(tx.to_json(orient="records", force_ascii=False)),
            "forecast": forecast, "risk": risk, "warnings": list(dict.fromkeys(warnings)),
            "complete_history": complete_history}
