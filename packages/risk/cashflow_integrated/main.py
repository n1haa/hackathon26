"""Runnable demonstration using the single-company C002.csv statement.

The balance forecast below is intentionally a synthetic fixture to demonstrate
risk_engine's interface: no forecasting implementation was supplied in this
repository. Replace `demo_forecast` with Vova's real forecast output for real use.
"""
from __future__ import annotations

from pathlib import Path
import argparse

if __package__:
    from .cashflow_pipeline import run_cashflow_pipeline
    from .data_loader import load_transactions
else:
    from cashflow_pipeline import run_cashflow_pipeline
    from data_loader import load_transactions


DEMO_FORECAST = [
    {"date": "2026-10-01", "forecast_balance": 320_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-02", "forecast_balance": 240_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-03", "forecast_balance": 100_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-04", "forecast_balance": -25_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-05", "forecast_balance": -80_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-06", "forecast_balance": -12_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-07", "forecast_balance": 40_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-08", "forecast_balance": 25_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-09", "forecast_balance": 15_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-10", "forecast_balance": 5_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-11", "forecast_balance": 2_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-12", "forecast_balance": 1_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-13", "forecast_balance": 0, "current_balance": 3_027_511.87},
    {"date": "2026-10-14", "forecast_balance": 150_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-15", "forecast_balance": 125_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-16", "forecast_balance": 80_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-17", "forecast_balance": 40_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-18", "forecast_balance": 35_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-19", "forecast_balance": 20_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-20", "forecast_balance": 10_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-21", "forecast_balance": 5_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-22", "forecast_balance": 3_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-23", "forecast_balance": 2_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-24", "forecast_balance": 1_000, "current_balance": 3_027_511.87},
    {"date": "2026-10-25", "forecast_balance": 700, "current_balance": 3_027_511.87},
    {"date": "2026-10-26", "forecast_balance": 400, "current_balance": 3_027_511.87},
    {"date": "2026-10-27", "forecast_balance": 250, "current_balance": 3_027_511.87},
    {"date": "2026-10-28", "forecast_balance": 100, "current_balance": 3_027_511.87},
    {"date": "2026-10-29", "forecast_balance": 50, "current_balance": 3_027_511.87},
    {"date": "2026-10-30", "forecast_balance": 25, "current_balance": 3_027_511.87},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo CashFlow AI module integration")
    parser.add_argument(
        "--transactions", default=str(Path(__file__).with_name("C002.csv")),
        help="CSV одной компании (UTF-8 BOM, разделитель ;)"
    )
    parser.add_argument("--company-id", help="Компания для выбора из общего CSV")
    args = parser.parse_args()

    path = Path(args.transactions)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Передайте путь, например --transactions /path/to/C002.csv"
        )

    transactions = load_transactions(path, company_id=args.company_id)

    # No forecasting module was present in the supplied repository. This fixture
    # makes the joined API runnable; it must not be presented as a real prediction.
    result = run_cashflow_pipeline(
        transactions,
        forecast=DEMO_FORECAST,
        forecast_start_date="2026-10-01",
        horizon_days=30,
    )

    print(f"Компания: {result['company_id']}")
    print(f"Исторических операций: {len(transactions)}")
    print(f"Обнаружено шаблонов регулярности: {len(result['recurring_events'])}")
    print(f"Ожидаемых событий для окна прогноза: {len(result['forecast_inputs'])}")
    print("\nПримеры ожидаемых событий (формат для forecasting):")
    for event in result["forecast_inputs"][:5]:
        print(
            f"  {event['date']} | {event['direction']} | "
            f"{event['amount']:,.2f} | {event['counterparty']} | "
            f"confidence={event['confidence']:.3f}"
        )

    risk = result["cash_gap"]
    print("\nКассовые риски на переданном демонстрационном прогнозе:")
    print(f"  Статус: {risk['status']}")
    print(f"  Минимальный остаток: {risk['minimum_balance']}")
    print(f"  Максимальный дефицит: {risk['maximum_deficit']}")
    for event in risk["events"]:
        print(
            f"  {event['start_date']}—{event['end_date']}: "
            f"дефицит {event['maximum_deficit']:,.2f}, "
            f"{event['duration_days']} дн., риск={event['risk_level']}"
        )

    print("\nПредупреждения:")
    for warning in result["warnings"]:
        print(f"  - {warning}")
    print("\nВажно: DEMO_FORECAST — тестовый пример, не прогноз модели forecasting.")


if __name__ == "__main__":
    main()
