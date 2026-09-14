import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "data/configs/examples/hyy.config"


def test_static_evidence_exit_does_not_require_coffea_compatibility(tmp_path):
    config = tmp_path / "analysis.config"
    config.write_text(CONFIG.read_text().replace('Job: "hyy"', 'Job: "hyy"\n  ImageFormat: pdf'))
    command = [
        sys.executable, "-m", "trex_fitter.config_verify", str(config),
        "--actions", "n", "--evidence-level", "S",
    ]
    static = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True, check=False)
    full = subprocess.run(command[:-1] + ["C"], cwd=PROJECT, text=True, capture_output=True, check=False)
    assert static.returncode == 0
    assert "VALID:" in static.stdout and "INCOMPATIBLE:" in static.stdout
    assert full.returncode == 1


def test_input_evidence_requires_input_check_flag():
    process = subprocess.run(
        [sys.executable, "-m", "trex_fitter.config_verify", str(CONFIG),
         "--actions", "n", "--evidence-level", "I"],
        cwd=PROJECT, text=True, capture_output=True, check=False,
    )
    assert process.returncode == 2
    assert "requires --check-inputs" in process.stderr
