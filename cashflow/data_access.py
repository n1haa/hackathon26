"""Replaceable data boundary for future team integrations."""

from . import demo_data


def get_transactions():
    return demo_data.get_transactions()


def get_recurring_payments():
    return demo_data.get_recurring_payments()


def get_forecast(days: int = 30):
    return demo_data.get_forecast(days)


def get_risks():
    return demo_data.get_risks()


def get_financial_summary():
    return demo_data.get_financial_summary()
