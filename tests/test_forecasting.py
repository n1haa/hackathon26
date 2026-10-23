from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest

from forecasting import build_features, evaluate_models, forecast_cashflow, prepare_daily_cashflows
from forecasting.evaluation import calculate_metrics
from forecasting.models import predict_flows
from forecasting.service import ForecastWarning
from forecasting.synthetic import make_synthetic_transactions


def history(days=14, income=100, expense=40):
    records = []
    for date in pd.date_range("2026-01-01", periods=days):
        records.extend([{"date": date, "direction": "incoming", "amount": income, "category": "revenue"},
                        {"date": date, "direction": "outgoing", "amount": expense, "category": "purchases"}])
    return records


@pytest.mark.parametrize("horizon", [1, 7, 30, 45])
def test_horizon_and_exact_financial_identity(horizon):
    result = forecast_cashflow(history(income=10.13, expense=20.27), -5, horizon)
    assert result["horizon"] == horizon
    assert len(result["forecast"]) == horizon
    assert result["forecast"][0]["date"] == "2026-01-15"
    balance = Decimal("-5")
    for row in result["forecast"]:
        income, expense, net = [Decimal(str(row[k])) for k in ("predicted_income", "predicted_expense", "predicted_net_flow")]
        assert income >= 0 and expense >= 0
        assert net == income - expense
        balance += net
        assert Decimal(str(row["predicted_balance"])) == balance
    json.dumps(result, allow_nan=False)


def test_single_day_uses_honest_baseline_without_measured_quality():
    result = forecast_cashflow(history(1), 0)
    assert result["model_name"] == "weekday_baseline"
    assert result["metrics"]["status"] == "insufficient_history"
    assert result["metrics"]["test"] == {}
    assert all(r["predicted_income"] == 100 for r in result["forecast"])
    assert any("Less than one week" in w for w in result["warnings"])


def test_zero_flows_and_negative_balance_are_supported():
    result = forecast_cashflow(history(3, 0, 0), -100)
    assert all(r["predicted_balance"] == -100 for r in result["forecast"])
    assert all(r["predicted_net_flow"] == 0 for r in result["forecast"])


def test_forecast_and_evaluation_round_half_cents_the_same_way():
    from forecasting.evaluation import _rounded_flows
    from forecasting.data import rounded_prediction
    assert rounded_prediction(0.005) == Decimal("0.01")
    assert _rounded_flows(pd.DataFrame({"income": [0.005], "expense": [0.015]})).iloc[0].tolist() == [.01, .02]
    records = history(14, income=0, expense=0)
    for record in records[14:]:
        if record["direction"] == "incoming":
            record["amount"] = .01
    assert forecast_cashflow(records, 0, horizon=1)["forecast"][0]["predicted_income"] == .01


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), True, "1.001", None])
def test_invalid_amount_rejected(amount):
    with pytest.raises(ValueError, match="amount"):
        forecast_cashflow(history(1, amount), 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad"])
def test_invalid_current_balance_rejected(value):
    with pytest.raises(ValueError, match="current_balance"):
        forecast_cashflow(history(1), value)


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        forecast_cashflow(history(1), 0, horizon)


def test_missing_coverage_is_nan_and_forecast_refuses_it():
    records = [history(3)[0], history(3)[-1]]
    prepared = prepare_daily_cashflows(records)
    assert pd.isna(prepared.daily.loc["2026-01-02", "income"])
    assert not prepared.daily.loc["2026-01-02", "observed"]
    with pytest.raises(ValueError, match="coverage"):
        forecast_cashflow(records, 0)
    result = forecast_cashflow(records, 0, complete_history=True)
    assert len(result["forecast"]) == 30


def test_explicit_coverage_includes_idle_days_at_both_ends():
    records = [{"date": "2026-01-02", "direction": "incoming", "amount": 20}]
    prepared = prepare_daily_cashflows(records, history_start="2026-01-01", as_of="2026-01-04",
                                       observed_dates=pd.date_range("2026-01-01", "2026-01-04"))
    assert prepared.daily["income"].tolist() == [0, 20, 0, 0]
    assert prepared.daily["expense"].tolist() == [0, 0, 0, 0]
    result = forecast_cashflow(records, 20, as_of="2026-01-04", complete_history=True)
    assert result["forecast"][0]["date"] == "2026-01-05"


def test_transaction_dates_must_be_in_explicit_coverage():
    with pytest.raises(ValueError, match="observed_dates"):
        prepare_daily_cashflows(history(2), observed_dates=["2026-01-01"])


def test_sorting_russian_directions_and_field_adapter_without_input_mutation():
    frame = pd.DataFrame([{"Дата": "2026-01-02", "Сумма": "5", "Направление": "списание"},
                          {"Дата": "2026-01-01", "Сумма": "10", "Направление": "поступление"}])
    before = frame.copy(deep=True)
    prepared = prepare_daily_cashflows(frame, fields={"date": "Дата", "amount": "Сумма", "direction": "Направление"})
    assert prepared.daily["net_flow"].tolist() == [10, -5]
    pd.testing.assert_frame_equal(frame, before)


def test_internal_transfers_do_not_change_consolidated_forecast():
    records = [dict(date="2026-01-01T10:00", amount=10, direction="outgoing", account_id="a",
                    category="internal_transfer", closing_balance=90),
               dict(date="2026-01-01T10:01", amount=10, direction="incoming", account_id="b",
                    category="internal_transfer", closing_balance=60)]
    result = forecast_cashflow(records, 150)
    assert all(r["predicted_balance"] == 150 for r in result["forecast"])
    assert all(r["predicted_income"] == r["predicted_expense"] == 0 for r in result["forecast"])
    assert not any("mismatch" in w for w in result["warnings"])


def test_mismatched_balances_warn_and_keep_explicit_current_anchor():
    records = [dict(date="2026-01-01", amount=10, direction="incoming", closing_balance=1000)]
    with pytest.warns(ForecastWarning):
        result = forecast_cashflow(records, 50, opening_balance=100, horizon=1)
    assert result["forecast"][0]["predicted_balance"] == 60
    assert any("operations=110" in w for w in result["warnings"])
    assert any("current_balance mismatch" in w for w in result["warnings"])


def test_one_cent_discrepancy_is_not_hidden():
    prepared = prepare_daily_cashflows([dict(date="2026-01-01", amount=10, direction="incoming", closing_balance="110.01")],
                                       opening_balance=100)
    assert any("difference=0.01" in w for w in prepared.warnings)


def test_sparse_account_snapshots_reconcile_intervening_transactions():
    records = [dict(date="2026-01-01", amount=10, direction="incoming", closing_balance=110),
               dict(date="2026-01-02", amount=20, direction="outgoing"),
               dict(date="2026-01-03", amount=30, direction="incoming", closing_balance=120)]
    prepared = prepare_daily_cashflows(records, opening_balance=100, current_balance=120)
    assert prepared.warnings == []


def test_reject_future_data_and_convert_timezone_before_cutoff():
    record = dict(date="2026-01-01T22:00:00Z", amount=1, direction="incoming")
    prepared = prepare_daily_cashflows([record], as_of="2026-01-02")
    assert prepared.as_of == pd.Timestamp("2026-01-02")
    with pytest.raises(ValueError, match="future"):
        forecast_cashflow([record], 1, as_of="2026-01-01")


@pytest.mark.parametrize("date", [None, "bad", 12, True, pd.NaT])
def test_invalid_dates(date):
    with pytest.raises(ValueError, match="date"):
        forecast_cashflow([dict(date=date, amount=1, direction="incoming")], 0)


def test_empty_and_invalid_direction():
    with pytest.raises(ValueError, match="Nonempty"):
        forecast_cashflow([], 0)
    with pytest.raises(ValueError, match="direction"):
        forecast_cashflow([dict(date="2026-01-01", amount=1, direction="sideways")], 0)


@pytest.mark.parametrize("field", ["currency", "company_id", "transaction_id"])
def test_mixed_business_currency_and_duplicates_are_rejected(field):
    records = history(1)
    records[0][field] = "same"
    records[1][field] = "same" if field == "transaction_id" else "different"
    with pytest.raises(ValueError):
        forecast_cashflow(records, 0)


def test_features_are_causal_and_use_shift_before_rolling():
    daily = prepare_daily_cashflows(history(70)).daily
    before = build_features(daily)
    cutoff = daily.index[40]
    changed = daily.copy()
    changed.loc[cutoff:, [c for c in changed if c != "observed"]] = 999999
    after = build_features(changed)
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    assert before.loc[cutoff, "income_lag_1"] == 100
    assert before.loc[cutoff, "expense_sum_7"] == 280
    assert np.isnan(before.iloc[27]["income_mean_28"])
    pd.testing.assert_frame_equal(build_features(daily.iloc[:41]), before.iloc[:41])


def test_ml_requires_enough_data_and_produces_recursive_nonnegative_flows():
    dates = pd.date_range("2026-01-15", periods=2)
    with pytest.raises(ValueError, match="56"):
        predict_flows(prepare_daily_cashflows(history()).daily, dates, "hist_gradient_boosting")
    daily = prepare_daily_cashflows(history(60)).daily
    dates = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=3)
    predicted = predict_flows(daily, dates, "hist_gradient_boosting")
    assert predicted.shape == (3, 2)
    assert np.isfinite(predicted).all().all()
    assert (predicted >= 0).all().all()


def test_metrics_defined_correctly_and_zero_wape_is_null():
    actual = pd.DataFrame({"income": [0, 0], "expense": [2, 2]})
    predicted = pd.DataFrame({"income": [1, 1], "expense": [1, 3]})
    metric = calculate_metrics(actual, predicted)
    assert metric["income"] == {"mae": 1., "rmse": 1., "wape": None}
    assert metric["expense"]["wape"] == .5
    assert metric["balance_mae"] == 2.


def test_selection_uses_validation_even_when_ml_wins_terminal_test(monkeypatch):
    import forecasting.evaluation as evaluation
    daily = prepare_daily_cashflows(history(150, 10, 10)).daily
    daily.loc[daily.index[-30:], ["income", "expense"]] = 30
    calls = []
    def fake_predict(train, dates, name):
        assert train.index[-1] < dates[0]
        calls.append((train.index[-1], dates[0], dates[-1]))
        amount = {"weekday_baseline": 10, "exponential_smoothing": 20, "hist_gradient_boosting": 30}[name]
        return pd.DataFrame({"income": amount, "expense": amount}, index=dates)
    monkeypatch.setattr(evaluation, "predict_flows", fake_predict)
    report = evaluate_models(daily, 30)
    assert report["selected_model"] == "weekday_baseline"
    assert report["test"]["hist_gradient_boosting"]["flow_mae"] == 0
    assert report["test"]["weekday_baseline"]["flow_mae"] == 20
    assert len(report["folds"]) == 2
    assert len(calls) == 9
    assert report["folds"][-1]["validation_end"] < report["test_period"]["start"]


def test_real_model_comparison_has_three_models_and_serializable_metrics():
    transactions, metadata = make_synthetic_transactions(130)
    daily = prepare_daily_cashflows(transactions, complete_history=True).daily
    report = evaluate_models(daily, horizon=10, max_folds=2)
    assert report["status"] == "evaluated"
    assert set(report["validation"]) == set(report["test"]) == {"weekday_baseline", "exponential_smoothing", "hist_gradient_boosting"}
    assert report["failures"] == {}
    selected = report["selected_model"]
    assert report["validation"][selected]["flow_mae"] <= report["validation"]["weekday_baseline"]["flow_mae"]
    json.dumps(report, allow_nan=False)


def test_recurring_history_removed_before_schedule_added_once():
    records = history(14)
    for record in records:
        if record["direction"] == "outgoing":
            record.update(recurring_id="rent", category="rent")
    event = dict(event_id="rent-jan", recurring_id="rent", date="2026-01-15", direction="outgoing", amount=120)
    result = forecast_cashflow(records, 1000, recurring_events=[event])
    assert result["forecast"][0]["predicted_expense"] == 120
    assert result["forecast"][0]["model_expense"] == 0
    assert all(r["predicted_expense"] == 0 for r in result["forecast"][1:])
    assert result["metrics"]["scope"] == "residual_flows_without_recurring_series"


def test_new_event_is_explicit_and_duplicates_cannot_double_count():
    event = dict(event_id="new", recurring_id="new", date="2026-01-15", direction="outgoing", amount=20, is_new=True)
    result = forecast_cashflow(history(), 0, recurring_events=[event])
    assert result["forecast"][0]["predicted_expense"] == 60
    with pytest.raises(ValueError, match="Duplicate event_id"):
        forecast_cashflow(history(), 0, recurring_events=[event, event])
    with pytest.raises(ValueError, match="Duplicate recurring occurrence"):
        forecast_cashflow(history(), 0, recurring_events=[event, {**event, "event_id": "second"}])
    with pytest.raises(ValueError, match="match historical"):
        forecast_cashflow(history(), 0, recurring_events=[{**event, "is_new": False}])


@pytest.mark.parametrize("change", [{"date": "2026-01-14"}, {"date": "2027-01-01"}, {"amount": -1}, {"category": "internal_transfer"}])
def test_invalid_events_rejected(change):
    event = dict(event_id="new", recurring_id="new", date="2026-01-15", direction="outgoing", amount=20, is_new=True)
    with pytest.raises(ValueError):
        forecast_cashflow(history(), 0, recurring_events=[{**event, **change}])


def test_partial_recurring_schedule_does_not_silently_erase_history():
    records = history()
    records[0]["recurring_id"] = "sales"
    records[1]["recurring_id"] = "rent"
    event = dict(event_id="rent-jan", recurring_id="rent", date="2026-01-15", direction="outgoing", amount=120)
    with pytest.raises(ValueError, match="complete horizon schedule"):
        forecast_cashflow(records, 0, recurring_events=[event])


def test_synthetic_data_balances_reconcile_and_source_is_explicit():
    records, metadata = make_synthetic_transactions()
    assert all(r["synthetic"] for r in records)
    assert metadata["synthetic"]
    prepared = prepare_daily_cashflows(records, opening_balance=metadata["opening_balance"],
                                       current_balance=metadata["current_balance"], complete_history=True)
    assert prepared.warnings == []
