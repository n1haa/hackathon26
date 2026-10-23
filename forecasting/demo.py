"""python -m forecasting.demo --output artifacts/synthetic_demo"""
import argparse
import json
from pathlib import Path

from .service import forecast_cashflow
from .synthetic import make_synthetic_transactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_demo"))
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    transactions, metadata = make_synthetic_transactions(args.days, args.seed)
    result = forecast_cashflow(transactions, metadata["current_balance"], args.horizon,
                               history_start=metadata["history_start"], as_of=metadata["as_of"],
                               opening_balance=metadata["opening_balance"], complete_history=True)
    result["data_source"] = "SYNTHETIC: fictional business; not evidence of accuracy on bank data"
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {"transactions.json": transactions, "metadata.json": metadata,
                 "forecast.json": result, "evaluation.json": {"data_source": result["data_source"], **result["metrics"]}}
    for filename, value in artifacts.items():
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"model": result["model_name"], "data_source": result["data_source"],
                      "test": result["metrics"]["test"], "artifacts": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
