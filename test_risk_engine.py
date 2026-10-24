import pandas as pd
import pytest

from risk_engine import RiskThresholds, analyze_cash_gap, detect_cash_gaps


def rows(balances, start="2026-01-01", dates=None, current_balance=1000):
    if dates is None:
        dates = pd.date_range(start, periods=len(balances), freq="D")
    return pd.DataFrame({
        "date": dates,
        "forecast_balance": balances,
        "current_balance": [current_balance] * len(balances),
    })


def test_no_cash_gap():
    result = analyze_cash_gap(rows([100, 50, 0, 10]))
    assert result["events"] == []
    assert result["status"] == "no_gap"
    assert result["minimum_balance"] == 0
    assert result["maximum_deficit"] == 0
    assert result["probability"] is None


def test_one_day_negative_balance():
    result = analyze_cash_gap(rows([20, -25, 30]))
    event = result["events"][0]
    assert event["start_date"] == "2026-01-02"
    assert event["end_date"] == "2026-01-02"
    assert event["minimum_balance"] == -25
    assert event["maximum_deficit"] == 25
    assert event["duration_days"] == 1
    assert event["risk_level"] == "low"


def test_several_consecutive_negative_days():
    result = analyze_cash_gap(rows([10, -20, -80, -5, 100]))
    event = result["events"][0]
    assert event["start_date"] == "2026-01-02"
    assert event["end_date"] == "2026-01-04"
    assert event["minimum_balance"] == -80
    assert event["maximum_deficit"] == 80
    assert event["duration_days"] == 3
    assert event["risk_level"] == "medium"


def test_multiple_separate_gaps():
    result = analyze_cash_gap(rows([-10, -20, 5, 0, -3, -4, 8]))
    assert len(result["events"]) == 2
    assert (result["events"][0]["start_date"],
            result["events"][0]["end_date"]) == (
                "2026-01-01", "2026-01-02")
    assert (result["events"][1]["start_date"],
            result["events"][1]["end_date"]) == (
                "2026-01-05", "2026-01-06")


def test_positive_balance_restoration_closes_event():
    result = analyze_cash_gap(rows([-10, -5, 0, -1]))
    assert len(result["events"]) == 2
    assert result["events"][0]["end_date"] == "2026-01-02"
    assert result["events"][1]["start_date"] == "2026-01-04"


def test_zero_balance_is_not_negative():
    assert detect_cash_gaps(rows([0, 0, 10])) == []


def test_missing_dates_warn_and_split_events():
    dates = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]
    result = analyze_cash_gap(rows([-10, -20, -30, -40], dates=dates))
    assert len(result["events"]) == 2
    assert any("Прогноз неполный" in warning for warning in result["warnings"])
    assert result["events"][0]["end_date"] == "2026-01-02"
    assert result["events"][1]["start_date"] == "2026-01-05"


def test_invalid_rows_are_skipped_with_warning():
    frame = pd.DataFrame({
        "date": ["2026-01-01", "not-a-date", "2026-01-03"],
        "forecast_balance": [-5, -100, 10],
        "current_balance": [100, 100, 100],
    })
    result = analyze_cash_gap(frame)
    assert len(result["events"]) == 1
    assert result["events"][0]["start_date"] == "2026-01-01"
    assert any("некорректных строк" in warning for warning in result["warnings"])


def test_empty_forecast():
    result = analyze_cash_gap([])
    assert result["events"] == []
    assert result["status"] == "unknown"
    assert result["minimum_balance"] is None
    assert result["probability"] is None
    assert result["warnings"]


def test_negative_balance_at_start_of_horizon():
    result = analyze_cash_gap(rows([-50, -20, 100]))
    event = result["events"][0]
    assert event["start_date"] == "2026-01-01"
    assert event["end_date"] == "2026-01-02"
    assert event["maximum_deficit"] == 50


def test_missing_balance_column_does_not_reconstruct_cashflows():
    result = analyze_cash_gap([{"date": "2026-01-01", "amount": 100}])
    assert result["events"] == []
    assert result["status"] == "unknown"
    assert any("Не найден прогнозируемый остаток" in w for w in result["warnings"])


def test_duplicate_dates_stop_ambiguous_analysis():
    result = analyze_cash_gap(rows([-10, 10], dates=["2026-01-01", "2026-01-01"]))
    assert result["events"] == []
    assert result["status"] == "unknown"
    assert any("повторяющиеся даты" in w for w in result["warnings"])


def test_configurable_risk_thresholds():
    custom = RiskThresholds(
        medium_deficit_ratio=0.05,
        high_deficit_ratio=0.10,
        medium_duration_days=2,
        high_duration_days=4,
    )
    result = analyze_cash_gap(rows([-200, -200], current_balance=1000),
                              thresholds=custom)
    assert result["events"][0]["risk_level"] == "high"


def test_dict_envelope_input():
    result = analyze_cash_gap({
        "forecast": [
            {"date": "2026-02-01", "forecast_balance": -20, "current_balance": 100},
            {"date": "2026-02-02", "forecast_balance": 10, "current_balance": 100},
        ]
    })
    assert len(result["events"]) == 1
    assert result["events"][0]["maximum_deficit"] == 20
