# recurring_detector

## Dataset findings
- `all_operations.csv`: 222,420 rows; columns match the supplied README.
- Direction values are `Поступление` and `Списание`.
- Dataset has no `category` column. `category` is optional in the module and returned as `None` when not supplied; categories are not invented.
- Input is UTF-8 with BOM and semicolon-delimited. Example load:

```python
import pandas as pd
from recurring_detector import detect_recurring

transactions = pd.read_csv(
    "all_operations.csv",
    sep=";",
    encoding="utf-8-sig",
    dtype={"company_id": str, "operation_id": str, "counterparty_inn": str},
)
events = detect_recurring(transactions)
```

## Output
`detect_recurring(transactions)` returns a list of dictionaries containing:
`company_id`, `counterparty`, `category`, `direction`, `frequency`, `median_amount`,
`expected_amount`, `expected_next_date`, `interval_days`, `occurrences_count`,
`confidence`, `source`.

`company_id` is included to identify which company owns the event in the multi-company file. `confidence` is a heuristic score for evidence of recurrence, not a probability
that the next payment will happen. `source="history"` means it was inferred
from historical transactions.

## Grouping and limitations
- Transactions are never grouped across companies or directions.
- Counterparty INN is preferred, then counterparty name. If both are unavailable,
  normalized payment-purpose text is used only when non-empty.
- If an upstream `category` field exists it is used. The supplied dataset has no
  category field, so the detector uses normalized payment purpose as a conservative
  sub-grouping key instead of inventing categories.
- Same-day operations are aggregated before interval analysis.
- Monthly dates are calendar-aware and month-end days are clamped. Expected dates
  falling on weekends are moved to the previous weekday. Public holidays are not
  adjusted because no production calendar is provided.
- Confidence is a transparent heuristic; tune the threshold against a labeled
  validation set before treating it as a production decision rule.
- The detector intentionally does not infer a payment probability.
