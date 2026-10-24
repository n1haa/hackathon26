"""Portable root commands: no activated environment or Make required."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*args):
    subprocess.run([str(PYTHON), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "build", "test", "lint", "run", "check"])
    args, extra = parser.parse_known_args()
    if args.command == "install":
        if not PYTHON.exists():
            venv.EnvBuilder(with_pip=True).create(ENV)
        lock = ROOT / "requirements.lock"
        if lock.exists():
            run("-m", "pip", "install", "-r", str(lock))
            run("-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ".")
        else:
            run("-m", "pip", "install", "-e", ".[dev]")
        run("-m", "pip", "check")
        return
    if not PYTHON.exists():
        parser.error("Run python manage.py install first")
    commands = {
        "build": ("-m", "build", "--no-isolation"),
        "test": ("-m", "pytest", "-q"),
        "lint": ("-m", "ruff", "check", "."),
        "run": ("-m", "cashflow_app.cli", "run"),
    }
    if args.command == "check":
        run("-m", "pip", "check")
        for command in ("lint", "test", "build"):
            run(*commands[command])
    else:
        run(*commands[args.command], *extra)


if __name__ == "__main__":
    main()
