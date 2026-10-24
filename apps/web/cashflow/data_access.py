"""Keep the existing UI schema while routing uploaded statements to real modules."""
import pandas as pd
import streamlit as st

from cashflow_app import analyze_statement
from . import demo_data

CATEGORIES = {"revenue": "Выручка", "other_income": "Прочие поступления", "purchases": "Закупки",
              "salary": "Зарплата", "rent": "Аренда", "taxes": "Налоги", "bank_fees": "Комиссии",
              "other_expense": "Прочие расходы", "internal_transfer": "Внутренний перевод", "unknown": "Не определена"}


def is_ready():
    return "analysis_input" in st.session_state


def result(days=30):
    if not is_ready():
        raise ValueError("Сначала выполните анализ на странице загрузки")
    cache = st.session_state.setdefault("analysis_results", {})
    if days not in cache:
        inputs = st.session_state.analysis_input
        cache[days] = analyze_statement(inputs["frame"], **inputs["options"], horizon=days)
    return cache[days]


def get_transactions():
    if not st.session_state.get("has_upload"):
        return demo_data.get_transactions()
    tx = pd.DataFrame(result()["transactions"])
    return pd.DataFrame({
        "Дата": pd.to_datetime(tx.operation_date), "Операция": tx.get("payment_purpose", "—"),
        "Контрагент": tx.get("counterparty_name", "—"), "Тип": tx.direction,
        "Категория": tx.category.map(CATEGORIES).fillna(tx.category),
        "Сумма": tx.amount.where(tx.direction.eq("Поступление"), -tx.amount),
        "Статус": "Обработано", "Остаток на конец дня": tx.get("balance_eod", "—"),
    }).sort_values("Дата", ascending=False, kind="stable")


def get_recurring_payments():
    if not st.session_state.get("has_upload"):
        return demo_data.get_recurring_payments()
    events = result()["risk"]["recurring_events"]
    return pd.DataFrame([{
        "Контрагент": event["counterparty"], "Категория": CATEGORIES.get(event["category"], event["category"]),
        "Направление": event["direction"], "Сумма": event["expected_amount"],
        "Следующая дата": event["expected_next_date"], "Периодичность": event["frequency"],
        "Оценка регулярности": event["confidence"],
    } for event in events], columns=["Контрагент", "Категория", "Направление", "Сумма", "Следующая дата", "Периодичность", "Оценка регулярности"])


def get_forecast(days: int = 30):
    if not st.session_state.get("has_upload"):
        return demo_data.get_forecast(days)
    output = result(days)
    future = pd.DataFrame(output["forecast"]["forecast"]).rename(columns={"predicted_balance": "balance"})
    future["date"] = pd.to_datetime(future.date)
    future["kind"] = "Прогноз"
    tx = pd.DataFrame(output["transactions"])
    if "balance_eod" in tx:
        history = tx.groupby("operation_date", as_index=False).balance_eod.first().rename(columns={"operation_date": "date", "balance_eod": "balance"})
        history["date"] = pd.to_datetime(history.date)
        history["kind"] = "Факт"
        return pd.concat([history.tail(30), future], ignore_index=True)
    return future


def get_risks():
    if not st.session_state.get("has_upload"):
        return demo_data.get_risks()
    return [{"title": "Ожидаемый кассовый разрыв", "date": event["start_date"],
             "gap_amount": event["maximum_deficit"], "amount": f'{event["maximum_deficit"]:,.2f} ₽',
             "minimum": f'{event["minimum_balance"]:,.2f} ₽', "level": event["risk_level"],
             "duration": f'{event["duration_days"]} дн.', "detail": event["explanation"], "probability": None}
            for event in result()["risk"]["cash_gap"]["events"]]


def get_financial_summary():
    if not st.session_state.get("has_upload"):
        return demo_data.get_financial_summary()
    output = result()
    tx = get_transactions()
    return {"balance": output["forecast"]["current_balance"],
            "forecast_balance": output["forecast"]["forecast"][-1]["predicted_balance"],
            "income": tx.loc[tx["Сумма"] > 0, "Сумма"].sum(),
            "expenses": -tx.loc[tx["Сумма"] < 0, "Сумма"].sum()}
