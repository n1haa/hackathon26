from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest

from cashflow_app import analyze_statement, read_statement


@pytest.fixture
def statement():
    return pd.DataFrame([
        {"company_id": "C001", "operation_id": f"op-{i}", "operation_date": day,
         "amount": "100.00", "direction": "Списание", "payment_purpose": "Аренда офиса",
         "counterparty_inn": "0012345678", "counterparty_name": "Landlord",
         "balance_eod": str(500 - i * 100)}
        for i, day in enumerate(("2026-10-01", "2026-10-08", "2026-10-15"))
    ])


def test_full_pipeline_preserves_inputs_and_does_not_double_count(statement):
    before = statement.copy(deep=True)
    output = analyze_statement(statement, current_balance=300, complete_history=True, horizon=7)
    pd.testing.assert_frame_equal(before, statement)
    assert output["company_id"] == "C001"
    assert {r["category"] for r in output["transactions"]} == {"rent"}
    assert output["risk"]["recurring_events"]
    assert output["risk"]["forecast_inputs"]
    assert output["risk"]["forecast_coverage"]["complete"] is True
    assert len(output["forecast"]["forecast"]) == 7
    assert output["forecast"]["forecast"][0]["date"] == "2026-10-16"
    assert all(r["scheduled_expense"] == 0 for r in output["forecast"]["forecast"])
    assert output["risk"]["forecast_builder_used"] is False
    assert output["risk"]["probability"] is None
    json.dumps(output, allow_nan=False)


def test_unknown_coverage_is_not_silently_zero_filled(statement):
    with pytest.raises(ValueError):
        analyze_statement(statement, current_balance=300)


def test_forecast_gap_reaches_risk_engine(statement):
    output = analyze_statement(statement.drop(columns="balance_eod"), current_balance=1,
                               complete_history=True, horizon=30)
    assert output["risk"]["cash_gap"]["events"]
    assert output["risk"]["maximum_deficit"] > 0
    assert output["risk"]["minimum_balance"] == min(row["predicted_balance"] for row in output["forecast"]["forecast"])


@pytest.mark.parametrize("column", ["company_id", "currency", "account_id"])
def test_separate_owners_accounts_and_currencies(statement, column):
    statement[column] = ["one", "two", "one"]
    with pytest.raises(ValueError, match=column):
        analyze_statement(statement, current_balance=300, complete_history=True)


def test_company_selection(statement):
    mixed = pd.concat([statement, statement.assign(company_id="C002")])
    output = analyze_statement(mixed, company_id="C002", current_balance=300, complete_history=True, horizon=7)
    assert {row["company_id"] for row in output["transactions"]} == {"C002"}


def test_inconsistent_eod_is_rejected(statement):
    statement.loc[1, "operation_date"] = statement.loc[0, "operation_date"]
    with pytest.raises(ValueError, match="balance_eod"):
        analyze_statement(statement, current_balance=300, complete_history=True)


def test_wrong_cutoff_balance_is_rejected(statement):
    with pytest.raises(ValueError, match="balance_eod"):
        analyze_statement(statement, current_balance=999, complete_history=True)


@pytest.mark.parametrize("amount", [True, "1.001", "nan", "-1"])
def test_invalid_money_is_rejected_before_coercion(statement, amount):
    statement.loc[0, "amount"] = amount
    with pytest.raises(ValueError):
        analyze_statement(statement, current_balance=300, complete_history=True)


@pytest.mark.parametrize("encoding,separator", [("utf-8-sig", ";"), ("cp1251", ";"), ("utf-8", ",")])
def test_csv_keeps_identifiers(statement, encoding, separator):
    data = statement.to_csv(index=False, sep=separator).encode(encoding)
    result = read_statement(data, filename="statement.csv")
    assert result.counterparty_inn.iloc[0] == "0012345678"
    assert len(result) == 3


def test_xlsx_keeps_identifiers(statement):
    buffer = BytesIO()
    statement.to_excel(buffer, index=False)
    assert read_statement(buffer, filename="statement.xlsx").counterparty_inn.iloc[0] == "0012345678"


def test_compatibility_modules_share_one_implementation():
    import recurring_detector
    import risk_engine
    from cashflow_integrated import recurring_detector as canonical_recurring
    from cashflow_integrated import risk_engine as canonical_risk
    assert recurring_detector is canonical_recurring
    assert risk_engine is canonical_risk


@pytest.mark.parametrize("uploaded", [False, True])
def test_all_ui_pages(statement, uploaded):
    from streamlit.testing.v1 import AppTest
    import cashflow_web
    app = AppTest.from_file(str(Path(cashflow_web.__file__).with_name("app.py")), default_timeout=60)
    if uploaded:
        options = {"current_balance": 300, "complete_history": True}
        output = analyze_statement(statement, **options)
        app.session_state["has_upload"] = True
        app.session_state["analysis_input"] = {"frame": statement, "options": options}
        app.session_state["analysis_results"] = {30: output}
        app.session_state["upload_frame"] = statement
    for page in ("Обзор", "Загрузка данных", "Операции", "Регулярные платежи", "Финансовый прогноз", "Финансовые риски"):
        app.session_state["page"] = page
        app.run()
        assert not app.exception, f"{page}: {app.exception}"
        assert not app.error, f"{page}: {app.error}"
        if uploaded:
            assert not any("ДЕМО-ДАННЫЕ" in node.value for node in app.markdown)
    if uploaded:
        app.session_state["page"] = "Финансовый прогноз"
        app.run()
        for period in ("7 дней", "90 дней"):
            app.radio(key="forecast_period").set_value(period).run()
            assert not app.exception
            assert not app.error


def test_upload_form_runs_real_analysis(statement):
    from streamlit.testing.v1 import AppTest
    import cashflow_web
    app = AppTest.from_file(str(Path(cashflow_web.__file__).with_name("app.py")), default_timeout=60)
    app.session_state["page"] = "Загрузка данных"
    app.session_state["has_upload"] = True
    app.session_state["upload_frame"] = statement
    app.run()
    app.checkbox[0].check()
    next(button for button in app.button if button.label == "Рассчитать прогноз и риски").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["analysis_results"][30]["risk"]["forecast_coverage"]["complete"]
    next(button for button in app.button if button.label == "Удалить выбранный файл").click().run()
    assert not app.exception
    assert "analysis_input" not in app.session_state
