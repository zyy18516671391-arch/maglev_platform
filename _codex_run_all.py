import ast
import subprocess
import sys
from pathlib import Path


PYTHON_ROOTS = ["core", "services", "training", "ui", "tests"]
EXTRA_FILES = ["_codex_run_all.py"]


def iter_python_files():
    for root in PYTHON_ROOTS:
        yield from Path(root).rglob("*.py")
    for file_name in EXTRA_FILES:
        yield Path(file_name)


def check_ast():
    for path in iter_python_files():
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        print(f"AST_OK {path}")


def run_pytest():
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    check_ast()
    run_pytest()
    print("ALL_OK")
