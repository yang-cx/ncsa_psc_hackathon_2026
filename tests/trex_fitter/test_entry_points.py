"""Compatibility checks for the package/script layout."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

from trex_fitter import runtime
from trex_fitter.config_format import ConfigError, _parse_blocks
from trex_fitter.coffea_backend import config


ROOT = Path(__file__).resolve().parents[2]


def test_shared_parser_retains_old_imports():
    assert config.ConfigError is ConfigError
    assert config._parse_blocks is _parse_blocks


def test_runtime_wrapper_is_same_module():
    legacy = importlib.import_module("trex_fitter.scripts.trex")
    assert legacy is runtime
    assert legacy.IMAGE == runtime.IMAGE


def test_evaluation_module_matches_script():
    commands = [[sys.executable, "-m", "trex_fitter.evaluate"],
                [sys.executable, "trex_fitter/scripts/evaluate_config.py"]]
    reports = []
    for command in commands:
        completed = subprocess.run(command + ["data/configs/examples/hyy.config", "--mock"],
                                   cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        report.pop("elapsed_seconds")
        reports.append(report)
    assert reports[0] == reports[1]
