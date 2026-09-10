"""Regression cases discovered in the fork review."""

import json
import subprocess
import sys
from pathlib import Path

import awkward as ak
import pytest

from trex_fitter.config_verify import verify_config
from trex_fitter.schema import load_schema, matches_schema
from trex_fitter.coffea_backend.expressions import Expression, boolean_mask


ROOT = Path(__file__).resolve().parents[2]
HYY = ROOT / "data/configs/examples/hyy.config"


@pytest.mark.parametrize("old,new", [
    ("UseMCstat: FALSE", "UseMCstat: BANANA"),
    ("DebugLevel: 1", "DebugLevel: one"),
    ("CorrelationThreshold: 0.2", "CorrelationThreshold: nope"),
    ("FitType: SPLUSB", "FitType: UNKNOWN"),
    ("FitRegion: CRSR", "FitRegion: CRSR\n  FitBlind: BANANA"),
    ("FillColor: 616", "FillColor: 616\n  FillColorRGB: 1,2"),
])
def test_invalid_native_setting_values(tmp_path, old, new):
    config = tmp_path / "invalid.config"
    config.write_text(HYY.read_text().replace(old, new))
    report = verify_config(config)
    assert not report.analysis_valid
    assert any(issue.code == "invalid_value" for issue in report.errors)


def test_native_allowed_setting_is_not_confused_with_backend_support(tmp_path):
    config = tmp_path / "native.config"
    config.write_text(HYY.read_text().replace("DebugLevel: 1", "DebugLevel: 1\n  StatOnly: TRUE"))
    report = verify_config(config)
    assert report.analysis_valid
    assert not report.coffea_compatible


def test_schema_alternatives_and_tuple_types():
    assert matches_schema("NONE", load_schema()["Job"]["MCstatThreshold"])
    assert matches_schema('"mu",1,0,10,TRUE', load_schema()["Sample"]["NormFactor"])
    assert not matches_schema("1,2,oops", load_schema()["Sample"]["FillColorRGB"])
    assert "MultiFit" in load_schema(True)


@pytest.mark.parametrize("source", ["pt[n]", "pt[-1]", "pt[1.5]", "pt[True]", "fabs(x, 2)", "atan2(x)", "sqrt()"])
def test_unsupported_expressions_fail_before_input_access(source):
    with pytest.raises(ValueError):
        Expression(source)


def test_boolean_guards_preserve_known_truth():
    events = ak.Array({"n": [0, 1, 1], "pt": [[], [50.], [20.]]})
    assert boolean_mask(Expression("n==0 || pt[0]>40")(events)).tolist() == [True, True, False]
    assert boolean_mask(Expression("!(n>0 && pt[0]>40)")(events)).tolist() == [True, False, True]
    assert boolean_mask(Expression("TRUE || pt[0]>40")(events)).tolist() == [True, True, True]
    assert boolean_mask(Expression("pt[0]>40")(events)).tolist() == [False, True, False]


def test_evaluator_mock_json():
    result = subprocess.run(
        [sys.executable, str(ROOT / "trex_fitter/scripts/evaluate_config.py"),
         str(HYY), "--mock"], cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["success"] and report["mock"]
    assert report["verification"]["analysis_valid"]
    assert report["significance"] == 1.7
