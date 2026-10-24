"""Reproducible local dataset audit: python -m cashflow_integrated.audit_data.

Writes measured data properties and a real event payload, not a synthetic
forecast. No probability or recurrence accuracy is inferred from unlabeled data.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import load_transactions, prepare_forecast_inputs


def profile(path: Path) -> dict:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig",
                        dtype={"company_id": str, "operation_id": str, "counterparty_inn": str})
    signed = frame.amount.where(frame.direction.eq("Поступление"), -frame.amount)
    daily = frame.assign(signed=signed).groupby(["company_id", "operation_date"]).agg(
        balance=("balance_eod", "last"), distinct_balances=("balance_eod", "nunique"), flow=("signed", "sum"),
    )
    discrepancy = daily.groupby(level=0).balance.diff() - daily.flow
    mismatches = discrepancy[discrepancy.abs().gt(0.011)]
    return {
        "file": path.name, "rows": len(frame), "companies": frame.company_id.nunique(),
        "date_start": frame.operation_date.min(), "date_end": frame.operation_date.max(),
        "columns": frame.columns.tolist(), "missing": frame.isna().sum().to_dict(),
        "directions": frame.direction.value_counts().to_dict(),
        "operations_per_company": frame.groupby("company_id").size().to_dict(),
        "duplicate_operation_ids": int(frame.duplicated(["company_id", "operation_id"]).sum()),
        "nonpositive_amounts": int(frame.amount.le(0).sum()),
        "negative_balance_days": int(daily.balance.lt(0).sum()),
        "conflicting_eod_days": int(daily.distinct_balances.gt(1).sum()),
        "reconciled_transitions_checked": int(discrepancy.notna().sum()),
        "inconsistent_balance_transitions_tolerance_0_011": int(discrepancy.abs().gt(0.011).sum()),
        "balance_mismatches": [
            {"company_id": company, "date": day, "balance_change_minus_flow": round(float(delta), 2)}
            for (company, day), delta in mismatches.items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = {"datasets": [profile(root / "all_operations.csv"), profile(root / "cashflow_integrated/C002.csv")]}
    payload = prepare_forecast_inputs(load_transactions(root / "cashflow_integrated/C002.csv"))
    report["c002_recurring"] = {
        "patterns": len(payload["recurring_events"]), "events_in_window": len(payload["events"]),
        "excluded_outside_window": payload["excluded_outside_window"],
        "start_date": payload["start_date"], "end_date": payload["end_date"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("data_audit.json", report), ("forecast_inputs_C002.json", payload)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
