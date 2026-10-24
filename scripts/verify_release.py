"""Install the release into a fresh venv and verify it outside the source tree."""
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "verification"
ENV = ROOT / ".venv-release"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(label, arguments, *, cwd=LOGS):
    with (LOGS / f"{label}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(PYTHON), *map(str, arguments)], cwd=cwd,
                                stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"{label}: exit={result.returncode}", flush=True)
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reinstall-wheel", action="store_true", help="Recheck a rebuilt wheel in the previously verified release environment")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if PYTHON.exists() and not args.reinstall_wheel:
        raise SystemExit("Use a fresh .venv-release directory; do not silently reuse an environment")
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=True).create(ENV)
        run("release-dependencies", ["-m", "pip", "install", "-r", ROOT / "requirements.lock"])
    wheel = ROOT / "dist" / "cashflow_workspace-0.1.0-py3-none-any.whl"
    run("release-wheel", ["-m", "pip", "install", "--no-deps", "--force-reinstall", wheel])
    run("release-pip-check", ["-m", "pip", "check"])
    run("release-smoke", ["-c", '''
import json
import subprocess
import sys
from pathlib import Path
import cashflow, cashflow_web, cashflow_app, forecasting, transaction_classifier, cashflow_integrated
from cashflow_app import read_statement
from streamlit.testing.v1 import AppTest
modules = [cashflow, cashflow_web, cashflow_app, forecasting, transaction_classifier, cashflow_integrated]
for module in modules:
    assert 'site-packages' in str(Path(module.__file__)), module.__file__
statement = Path(cashflow_integrated.__file__).with_name('C002.csv')
frame = read_statement(statement)
balance = frame.sort_values('operation_date').balance_eod.iloc[-1]
subprocess.run([sys.executable, '-m', 'cashflow_app.cli', 'analyze', str(statement),
                '--current-balance', str(balance), '--complete-history',
                '--output', 'sample-analysis.json'], check=True)
output = json.loads(Path('sample-analysis.json').read_text(encoding='utf-8'))
assert len(output['transactions']) == 1122
assert len(output['forecast']['forecast']) == 30
assert output['risk']['forecast_coverage']['complete']
json.dumps(output, allow_nan=False)
app = AppTest.from_file(str(Path(cashflow_web.__file__).with_name('app.py')), default_timeout=60)
app.run()
assert not app.exception and not app.error
print(json.dumps({'modules': {m.__name__: m.__file__ for m in modules},
                  'operations': len(output['transactions']), 'model': output['forecast']['model_name'],
                  'recurring_patterns': len(output['risk']['recurring_events']),
                  'risk_status': output['risk']['status'], 'forecast_days': len(output['forecast']['forecast'])}))
'''])
    run("release-cli", ["-m", "cashflow_app.cli", "--help"])
    app_path = subprocess.check_output([str(PYTHON), "-c", "import cashflow_web; from pathlib import Path; print(Path(cashflow_web.__file__).with_name('app.py'))"], cwd=LOGS, text=True).strip()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (LOGS / "release-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(PYTHON), "-m", "streamlit", "run", app_path,
                                    "--server.headless=true", f"--server.port={port}",
                                    "--server.address=127.0.0.1", "--browser.gatherUsageStats=false"],
                                   cwd=LOGS, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited before healthcheck")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        body = response.read().decode()
                        assert response.status == 200 and body == "ok", body
                    (LOGS / "release-health.json").write_text(json.dumps({"status": 200, "body": body}), encoding="utf-8")
                    print("HTTP health: 200 ok", flush=True)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Streamlit healthcheck timed out")
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
