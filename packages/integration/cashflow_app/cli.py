"""Installed CLI and Streamlit launcher, independent of the current directory."""
import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys

from .service import analyze_statement, read_statement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Start the Streamlit application")
    analyze = commands.add_parser("analyze", help="Analyze a CSV/XLSX statement")
    analyze.add_argument("statement", type=Path)
    analyze.add_argument("--company-id")
    analyze.add_argument("--current-balance", required=True)
    analyze.add_argument("--complete-history", action="store_true")
    analyze.add_argument("--own-inn")
    analyze.add_argument("--horizon", type=int, default=30)
    analyze.add_argument("--output", type=Path)
    args, extra = parser.parse_known_args()
    if args.command == "run":
        # Locate the packaged entry script without importing the UI into the service layer.
        app = Path(find_spec("cashflow_web").origin).with_name("app.py")
        defaults = ["--server.maxUploadSize=20", "--theme.primaryColor=#187a5b",
                    "--theme.backgroundColor=#f7f9f8", "--theme.secondaryBackgroundColor=#ffffff",
                    "--theme.textColor=#17211e", "--theme.font=sans serif"]
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *defaults, *extra]))
    if extra:
        parser.error(f"Unknown arguments: {' '.join(extra)}")
    try:
        result = analyze_statement(read_statement(args.statement), company_id=args.company_id,
                                   current_balance=args.current_balance, complete_history=args.complete_history,
                                   own_inn=args.own_inn, horizon=args.horizon)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка анализа: {exc}\n")
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
