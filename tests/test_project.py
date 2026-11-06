
import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_exists():
    assert (PROJECT_ROOT / "README.md").is_file()

def test_aequilibrae_importable():
    assert importlib.util.find_spec("aequilibrae") is not None, (
        "aequilibrae is not installed in this environment; run `uv sync`"
    )
    importlib.import_module("aequilibrae")

def test_aequilibrae_version():
    assert importlib.metadata.version("aequilibrae") >="1.7.2", ("aequilibrae version needs to be updated;")


def test_gurobi_license():
    gurobi_cl = shutil.which("gurobi_cl")
    assert gurobi_cl is not None, "gurobi_cl not found on PATH; install Gurobi"
    result = subprocess.run(
        [gurobi_cl, "--license"], capture_output=True, text=True, timeout=60
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0 and "error" not in output.lower(), (
        f"Gurobi license check failed:\n{output}"
    )
