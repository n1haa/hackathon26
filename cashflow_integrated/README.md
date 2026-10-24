# CashFlow AI — integration of `recurring_detector` and `risk_engine`

This package links the two supplied modules with `cashflow_pipeline.py` and a runnable `main.py` demonstration. The original modules were copied without changing their detection algorithms; only the file `risk_engine(1).py` was given the import-friendly name `risk_engine.py` inside this package.

## Important repository finding

The supplied files include `recurring_detector.py`, `risk_engine(1).py`, their test files, READMEs and **the statement of one business only: `C002.csv`**. There is no forecasting module or forecasting response schema in the provided files. Therefore the package does not invent or duplicate forecasting. It exposes an adapter contract so the existing forecast module can receive recurring events and return projected balances to `risk_engine`.

`C002.csv` is a single-company statement, not the combined dataset: **1,122 operations**, company `C002`, from `2024-10-01` through `2026-09-30`. The table has nine fields (`company_id`, `operation_id`, `operation_date`, `amount`, `direction`, `counterparty_inn`, `counterparty_name`, `payment_purpose`, `balance_eod`) and no category column. The integration validates that a single run does not combine different `company_id` values. INNs and IDs are read as strings to preserve leading zeros.

## Run the demonstration

Requirements: Python 3.10+, `pandas`, `numpy`, `pytest`.

```bash
pip install pandas numpy pytest
python main.py
pytest -q
```

Use another single-company statement with:

```bash
python main.py --transactions /path/to/company.csv
```

The demo runs recurrence detection on the included real `C002.csv`, then calls the risk engine on a **separate, explicitly illustrative forecast fixture** for 1–30 October 2026. The fixture is not a forecast generated from C002 data and must not be used for business decisions. It exists because the team's actual forecasting module was not included with the supplied files.

## Integrated API

```python
import pandas as pd
from cashflow_pipeline import run_cashflow_pipeline

transactions = pd.read_csv(
    "C002.csv",
    sep=";",
    encoding="utf-8-sig",
    dtype={"company_id": str, "operation_id": str, "counterparty_inn": str},
)

# `forecast_output` must be the real forecast result from Vova's module;
# its rows need a date and an already-computed balance column.
result = run_cashflow_pipeline(
    transactions,
    forecast=forecast_output,
)

recurring_events = result["recurring_events"]  # detector's complete history result
forecast_inputs = result["forecast_inputs"]    # expected events within forecast window
cash_gap = result["cash_gap"]                  # result from risk_engine
```

Top-level result fields:

- `company_id`: one company ID inferred from the input, e.g. `C002`.
- `recurring_events`: all detected recurring patterns based on available history.
- `forecast_inputs`: expected occurrences with `expected_next_date` falling inside the chosen forecast horizon, serialized for the forecast module.
- `forecast_window`: start, end and number of days.
- `cash_gap`: unchanged result shape from `analyze_cash_gap`.
- `status`, `minimum_balance`, `maximum_deficit`, `probability`, `probability_status`: convenience aliases from `cash_gap`.
- `warnings`: combined integration and risk-engine warnings.

### Event format for forecasting

Each item in `forecast_inputs` contains `date`, `direction`, `amount`, `signed_amount`, `counterparty`, `category`, `frequency`, `confidence`, `occurrences_count`, `company_id` and `source`. Incoming amounts have a positive `signed_amount`; outgoing amounts are negative. The forecasting module should use these event inputs as expected dated cash flows before producing the projected daily balance. The bridge does not add these amounts to the balance itself and does not create new occurrences beyond the detector's single `expected_next_date`.

### Connecting the actual forecasting function

If Vova's API can accept additional expected events, wrap it with `forecast_builder`. It receives keyword arguments `transactions`, `recurring_events`, `base_forecast`, `start_date`, `horizon_days`, and must return the final forecast in a format accepted by `risk_engine`:

```python
def forecasting_adapter(*, transactions, recurring_events, base_forecast,
                        start_date, horizon_days):
    # Adapt this body to the real forecasting API; do not copy the placeholder.
    return vova_forecasting_function(
        transactions=transactions,
        expected_events=recurring_events,
        start_date=start_date,
        horizon_days=horizon_days,
    )

result = run_cashflow_pipeline(
    transactions,
    forecast=None,
    forecast_builder=forecasting_adapter,
    forecast_start_date="2026-10-01",
    horizon_days=30,
)
```

The adapter example's `vova_forecasting_function` is a placeholder name, not an implementation shipped here. If the actual forecasting function has a different signature, this thin adapter is the only place that should translate to it. The forecasting module remains responsible for constructing balances; `risk_engine` only evaluates them.

If a forecast is provided without `forecast_builder`, `run_cashflow_pipeline` still runs the risk analysis but emits a warning that recurring signals have **not** been incorporated into that forecast. This prevents the output from implying a tighter integration than actually occurred.

## Forecast window and stale expected dates

The detector can return a next expected date before the new forecast starts if the last transaction for that specific counterparty/purpose was old. These events are not shifted forward, because doing so would invent dates not supported by the detector. Only events whose `expected_next_date` falls between the start date and the last day of the forecast window are passed to forecasting. Exclusions are reported in `warnings`.

The detector's `confidence` is a heuristic measure of regularity, **not the probability that a payment will happen**. `risk_engine.probability` remains `None`; risk levels (`low`, `medium`, `high`) are rule-based severity labels, not probabilities.

## Test coverage

The test run covers:

- all tests shipped with `recurring_detector` and `risk_engine`;
- joining the outputs, signed cash-flow conversion and forecast-window filtering;
- the forecast-builder callback contract;
- single-company input guard;
- invalid horizons and unknown directions;
- smoke test against the included `C002.csv` file.

## Findings and recommendations

1. **Agree on the actual forecast contract with Vova**: name of the daily balance field, date semantics and how recurring events should be provided to his module.
2. **The source has no category field**, so category is `None`. Do not invent categories in this bridge.
3. **The expected event list is not a probability model**. Consider using calibrated recurrence confidence only as a configurable input filter; validate it against labeled examples before relying on it.
4. **A stale next date is excluded rather than rolled forward**. A future enhancement can add explicit overdue-event logic, but it should require a business rule and evidence.
5. One account statement may not include all business liquidity. Other accounts, overdrafts and transfers between own accounts can affect the real cash position.
6. The included `main.py` demo fixture is synthetic; replace it with the actual forecasting model output before drawing conclusions about C002's future cash gaps.
