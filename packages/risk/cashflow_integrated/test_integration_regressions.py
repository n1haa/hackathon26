"""Regression and end-to-end tests for the task 2 -> forecasting -> task 4 contract."""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from . import load_transactions, prepare_forecast_inputs, run_cashflow_pipeline
from .cashflow_pipeline import prepare_recurring_events_for_forecast
from .data_loader import validate_transactions
from .test_cashflow_pipeline import history, small_forecast


@pytest.fixture(params=["", "cashflow_integrated."])
def detector(request):
    return importlib.import_module(request.param + "recurring_detector").detect_recurring


@pytest.fixture(params=["", "cashflow_integrated."])
def risk(request):
    return importlib.import_module(request.param + "risk_engine").analyze_cash_gap


def test_daily_payment_after_friday_is_in_the_future(detector):
    tx = history().iloc[:3].copy()
    tx["operation_date"] = ["2026-09-23", "2026-09-24", "2026-09-25"]
    assert detector(tx)[0]["expected_next_date"] == "2026-09-28"


def test_missing_inn_falls_back_to_separate_names(detector):
    first = history()
    first["counterparty_inn"] = pd.NA
    second = first.copy()
    second["counterparty_name"] = "Другой арендодатель"
    assert len(detector(pd.concat([first, second]))) == 2


def test_nonfinite_amount_cannot_become_expected_payment(detector):
    tx = history()
    tx["amount"] = float("inf")
    assert detector(tx) == []


@pytest.mark.xfail(strict=True, reason="Known detector quality limitation: irregular intervals pass the heuristic threshold; see AUDIT.md")
def test_irregular_negative_control_should_not_be_monthly(detector):
    tx = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-30", "2025-04-15", "2025-04-16", "2025-09-30"],
        "amount": [100] * 6, "direction": ["expense"] * 6, "counterparty": ["Supplier"] * 6,
    })
    assert detector(tx) == []


def test_categories_and_companies_are_not_merged(detector):
    first = history("C001")
    first["category"] = "rent"
    second = history("C002")
    second["category"] = "payroll"
    events = detector(pd.concat([first, second]))
    assert {(e["company_id"], e["category"]) for e in events} == {("C001", "rent"), ("C002", "payroll")}


def test_same_day_distinct_operations_are_summed(detector):
    tx = history()
    extra = tx.copy()
    extra["operation_id"] += "-second"
    event = detector(pd.concat([tx, extra]))[0]
    assert event["occurrences_count"] == 4
    assert event["expected_amount"] == 100_000


@pytest.mark.parametrize("bad_date", [pd.NaT, None, 12345])
def test_invalid_boundary_date_never_yields_no_gap(risk, bad_date):
    result = risk([{"date": "2026-10-01", "balance": 10}, {"date": bad_date, "balance": -20}])
    assert result["status"] == "incomplete"


def test_invalid_boundary_balance_never_yields_no_gap(risk):
    result = risk([{"date": "2026-10-01", "balance": 10}, {"date": "2026-10-02", "balance": float("inf")}])
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("horizon", [True, 1.5, "30", 0, -1, None])
def test_horizon_requires_positive_integer(horizon):
    with pytest.raises(ValueError, match="horizon_days"):
        prepare_forecast_inputs(history(), horizon_days=horizon)


def test_preparation_without_forecast_preserves_category_and_input():
    tx = history()
    tx["category"] = "rent"
    original = tx.copy(deep=True)
    payload = prepare_forecast_inputs(tx, start_date="2026-10-01")
    assert payload["end_date"] == "2026-10-30"
    assert payload["events"][0]["signed_amount"] == -50_000
    assert payload["events"][0]["category"] == "rent"
    json.dumps(payload, allow_nan=False)
    pd.testing.assert_frame_equal(tx, original)


def test_history_cannot_leak_into_forecast_window():
    with pytest.raises(ValueError, match="утечка"):
        prepare_forecast_inputs(history(), start_date="2026-08-01")


@pytest.mark.parametrize("field,value", [
    ("amount", -10), ("amount", float("inf")), ("amount", None),
    ("operation_date", "bad-date"), ("operation_date", 12345),
    ("direction", "UNKNOWN"), ("company_id", pd.NA),
])
def test_invalid_history_is_not_silently_discarded(field, value):
    tx = history().astype(object)
    tx.loc[0, field] = value
    with pytest.raises(ValueError):
        prepare_forecast_inputs(tx)


def test_duplicate_ids_rejected_but_identical_distinct_payments_retained():
    tx = history()
    with pytest.raises(ValueError, match="operation_id"):
        validate_transactions(pd.concat([tx, tx]))
    extra = tx.copy()
    extra["operation_id"] += "-second"
    assert len(validate_transactions(pd.concat([tx, extra]))) == 8


def test_loader_selects_company_and_preserves_leading_zeros(tmp_path):
    tx = pd.concat([history("C001"), history("C002")])
    path = tmp_path / "operations.csv"
    tx.to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    with pytest.raises(ValueError, match="company_id"):
        load_transactions(path)
    loaded = load_transactions(path, company_id="C002")
    assert loaded.counterparty_inn.tolist() == ["0012345678"] * 4
    with pytest.raises(ValueError, match="не найдена"):
        load_transactions(path, company_id="C999")


def test_spec_transaction_aliases_are_supported():
    tx = history().rename(columns={"operation_date": "date", "counterparty_inn": "inn",
                                   "counterparty_name": "counterparty", "payment_purpose": "description"})
    tx["direction"] = "expense"
    assert prepare_forecast_inputs(tx, start_date="2026-10-01")["events"][0]["signed_amount"] == -50_000


def test_short_positive_forecast_does_not_claim_safe_30_days():
    result = run_cashflow_pipeline(history(), forecast=[{"date": "2026-10-01", "balance": 100}])
    assert result["status"] == "incomplete"
    assert result["forecast_coverage"]["expected_days"] == 30
    assert len(result["forecast_coverage"]["missing_dates"]) == 29


@pytest.mark.parametrize("wrapped", [False, True])
def test_forecast_company_must_match_history(wrapped):
    forecast = [{"date": "2026-10-01", "balance": 100, "company_id": "C003"}]
    if wrapped:
        forecast = {"company_id": "C003", "forecast": [{"date": "2026-10-01", "balance": 100}]}
    with pytest.raises(ValueError, match="company_id"):
        run_cashflow_pipeline(history(), forecast=forecast)


def test_builder_cannot_return_dates_from_another_window():
    with pytest.raises(ValueError, match="окно"):
        run_cashflow_pipeline(history(), forecast_start_date="2026-10-01",
                              forecast_builder=lambda **kwargs: [{"date": "2027-01-01", "balance": -10}])


def test_builder_cannot_mutate_reported_inputs_or_callers_forecast():
    tx = history()
    forecast = small_forecast()
    original = json.dumps(forecast)

    def builder(**kwargs):
        kwargs["recurring_events"].clear()
        kwargs["base_forecast"][0]["forecast_balance"] = 1
        return kwargs["base_forecast"]

    result = run_cashflow_pipeline(tx, forecast=forecast, forecast_builder=builder)
    assert result["forecast_inputs"]
    assert json.dumps(forecast) == original


def test_end_to_end_events_change_balances_then_generate_gap():
    # An explicit test fixture, not a production forecasting model. It models
    # only the recurring component; there is no base flow to double count.
    def fixture_builder(*, transactions, recurring_events, base_forecast, start_date, horizon_days):
        balance = 40_000
        rows = []
        for day in pd.date_range(start_date, periods=horizon_days):
            day = day.date().isoformat()
            balance += sum(event["signed_amount"] for event in recurring_events if event["date"] == day)
            rows.append({"date": day, "balance": balance, "current_balance": 40_000})
        return {"forecast": rows}

    result = run_cashflow_pipeline(history(), forecast_builder=fixture_builder, forecast_start_date="2026-10-01")
    assert result["forecast_coverage"]["complete"]
    assert result["minimum_balance"] == -10_000
    gap = result["cash_gap"]["events"][0]
    assert (gap["start_date"], gap["end_date"], gap["duration_days"]) == ("2026-10-05", "2026-10-30", 26)
    assert gap["maximum_deficit"] == 10_000
    assert result["probability"] is None
    json.dumps(result, allow_nan=False)


def test_invalid_confidence_is_not_sent_to_forecasting():
    event = {"expected_next_date": "2026-10-05", "expected_amount": 100, "direction": "income", "confidence": 1.1}
    assert prepare_recurring_events_for_forecast([event], start_date="2026-10-01")["events"] == []


def test_supplied_c002_expected_events_are_serializable():
    tx = load_transactions(Path(__file__).with_name("C002.csv"))
    payload = prepare_forecast_inputs(tx)
    assert payload["company_id"] == "C002"
    assert len(payload["recurring_events"]) == 120
    assert len(payload["events"]) == 27
    assert all(event["category"] is None for event in payload["events"])
    json.dumps(payload, allow_nan=False)
