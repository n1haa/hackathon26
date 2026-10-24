import pandas as pd
import pytest

from cashflow_pipeline import (
    prepare_recurring_events_for_forecast,
    run_cashflow_pipeline,
)


def history(company="C002"):
    # Monthly recurring rent: next expected operation should fall on 2026-10-05.
    return pd.DataFrame({
        "company_id": [company] * 4,
        "operation_id": [f"{company}-{i}" for i in range(4)],
        "operation_date": ["2026-06-05", "2026-07-05", "2026-08-05", "2026-09-05"],
        "amount": [50_000, 50_000, 51_000, 50_000],
        "direction": ["Списание"] * 4,
        "counterparty_inn": ["0012345678"] * 4,
        "counterparty_name": ["ООО Арендодатель"] * 4,
        "payment_purpose": ["Арендная плата за помещение"] * 4,
    })


def small_forecast():
    return [
        {"date": "2026-10-01", "forecast_balance": 100_000, "current_balance": 100_000},
        {"date": "2026-10-02", "forecast_balance": -10_000, "current_balance": 100_000},
        {"date": "2026-10-03", "forecast_balance": -20_000, "current_balance": 100_000},
        {"date": "2026-10-04", "forecast_balance": 25_000, "current_balance": 100_000},
    ]


def test_prepare_recurring_event_has_signed_amount_and_window_filter():
    source = [
        {
            "company_id": "C002", "counterparty": "Rent Co", "category": None,
            "direction": "Списание", "frequency": "monthly",
            "median_amount": 50000, "expected_amount": 50000,
            "expected_next_date": "2026-10-05", "interval_days": 30,
            "occurrences_count": 12, "confidence": 0.91, "source": "history",
        },
        {
            "company_id": "C002", "counterparty": "Old Supplier", "category": None,
            "direction": "Списание", "frequency": "monthly",
            "median_amount": 1000, "expected_amount": 1000,
            "expected_next_date": "2025-01-01", "interval_days": 30,
            "occurrences_count": 12, "confidence": 0.95, "source": "history",
        },
    ]
    result = prepare_recurring_events_for_forecast(
        source, start_date="2026-10-01", horizon_days=30
    )
    assert len(result["events"]) == 1
    assert result["events"][0]["date"] == "2026-10-05"
    assert result["events"][0]["signed_amount"] == -50000
    assert result["excluded_outside_window"] == 1
    assert result["warnings"]


def test_integrated_pipeline_returns_recurring_and_cash_gap_results():
    result = run_cashflow_pipeline(
        history(), forecast=small_forecast(), forecast_start_date="2026-10-01", horizon_days=30
    )
    assert result["company_id"] == "C002"
    assert result["recurring_events"]
    assert result["forecast_inputs"]
    assert result["cash_gap"]["events"]
    assert result["cash_gap"]["minimum_balance"] == -20000
    assert result["probability"] is None
    assert any("не были встроены" in warning for warning in result["warnings"])


def test_forecast_builder_receives_only_in_horizon_recurring_events():
    captured = {}

    def builder(*, transactions, recurring_events, base_forecast, start_date, horizon_days):
        captured["company_ids"] = transactions["company_id"].unique().tolist()
        captured["events"] = recurring_events
        captured["start_date"] = start_date
        captured["horizon_days"] = horizon_days
        return base_forecast

    result = run_cashflow_pipeline(
        history(), forecast=small_forecast(), forecast_builder=builder,
        forecast_start_date="2026-10-01", horizon_days=30,
    )
    assert captured["company_ids"] == ["C002"]
    assert captured["start_date"] == "2026-10-01"
    assert captured["horizon_days"] == 30
    assert all("signed_amount" in event for event in captured["events"])
    assert result["forecast_builder_used"] is True
    assert not any("forecast_builder не передан" in warning for warning in result["warnings"])


def test_single_company_guard_prevents_mixing_forecasts():
    df = pd.concat([history("C002"), history("C003")], ignore_index=True)
    with pytest.raises(ValueError, match="одну компанию"):
        run_cashflow_pipeline(df, forecast=small_forecast())


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="horizon_days"):
        prepare_recurring_events_for_forecast([], start_date="2026-10-01", horizon_days=0)


def test_invalid_or_unknown_direction_does_not_get_signed_guess():
    event = {
        "company_id": "C002", "counterparty": "Unknown", "direction": "UNKNOWN",
        "expected_amount": 10, "expected_next_date": "2026-10-05", "confidence": 0.9,
    }
    result = prepare_recurring_events_for_forecast(
        [event], start_date="2026-10-01", horizon_days=30
    )
    assert result["events"] == []
    assert any("неизвестное направление" in warning for warning in result["warnings"])


def test_pipeline_requires_forecast_or_builder():
    with pytest.raises(ValueError, match="forecast или forecast_builder"):
        run_cashflow_pipeline(history())


def test_supplied_single_company_c002_dataset_smoke():
    from pathlib import Path
    from cashflow_pipeline import _validate_single_company
    from recurring_detector import detect_recurring

    csv_path = Path(__file__).with_name("C002.csv")
    df = pd.read_csv(
        csv_path, sep=";", encoding="utf-8-sig",
        dtype={"company_id": str, "operation_id": str, "counterparty_inn": str},
    )
    assert len(df) == 1122
    assert df["company_id"].nunique() == 1
    assert _validate_single_company(df) == "C002"
    events = detect_recurring(df)
    assert events
    assert {event["company_id"] for event in events} == {"C002"}
