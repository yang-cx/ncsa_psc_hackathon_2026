import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import uproot

from trex_fitter.config_verify import verify_config


ROOT = Path(__file__).resolve().parents[2]
HYY = ROOT / "data/configs/examples/hyy.config"


def edit(tmp_path, old, new):
    path = tmp_path / "test.config"
    path.write_text(HYY.read_text().replace(old, new))
    return path


@pytest.mark.parametrize("old,new,code", [
    ("POIAsimov: 1", "POIAsimov: wrong@1", "invalid_poi_asimov"),
    ("POIAsimov: 1", "POIAsimov: nan", "invalid_poi_asimov"),
    ("POIAsimov: 1", "POIAsimov: 1\n  NumCPU: 0", "numeric_range"),
    ("POIAsimov: 1", "POIAsimov: 1\n  ToysHistoNbins: 1", "numeric_range"),
    ("DebugLevel: 1", "DebugLevel: 1\n  RankingNPfraction: 1.1", "numeric_range"),
    ("DebugLevel: 1", 'DebugLevel: 1\n  RankingPOIName: "a","b"', "poi_list_length"),
    ('Sample: "Data"', 'Sample: "Data"\n  Regions: unknown', "unknown_region"),
])
def test_native_semantic_errors(tmp_path, old, new, code):
    report = verify_config(edit(tmp_path, old, new))
    assert not report.analysis_valid
    assert any(i.code == code and i.source_ref for i in report.errors)


def test_actions_and_validation_regions(tmp_path):
    path = edit(tmp_path, "Type: SIGNAL", "Type: VALIDATION")
    # Change region types only; signal sample types must stay valid.
    path.write_text(HYY.read_text().replace('  Type: SIGNAL\n  Variable:', '  Type: VALIDATION\n  Variable:'))
    assert verify_config(path, actions="n").analysis_valid
    assert not verify_config(path, actions="nwsf").analysis_valid
    assert not verify_config(path, actions=["n", "w", "s", "f"]).analysis_valid


@pytest.mark.parametrize("poi,addition", [
    ("alpha_lumi", '\nSystematic: "luminosity"\n  NuisanceParameter: lumi\n  Type: OVERALL\n  Samples: all\n'),
    ("inline_mu", '\nSample: "extra"\n  Type: SIGNAL\n  NtupleFiles: extra\n  NormFactor: inline_mu,1,0,10\n'),
    ("mass", '\nSample: "extra"\n  Type: GHOST\n  NtupleFiles: extra\n  Template: mass:125\n'),
    ("shape_mu", '\nShapeFactor: "shape"\n  Samples: ggH\n  Regions: cat_2jet\n  Expression: shape_mu:x\n'),
])
def test_poi_kinds(tmp_path, poi, addition):
    path = edit(tmp_path, 'POI: "mu_H"', f'POI: "{poi}"')
    path.write_text(path.read_text() + addition)
    report = verify_config(path)
    assert report.analysis_valid, report.errors


def test_duplicate_inline_attachment(tmp_path):
    path = edit(tmp_path, 'Sample: "ggH"', 'Sample: "ggH"\n  NormFactor: mu_H,1,0,50')
    assert any(i.code == "duplicate_normfactor_attachment" for i in verify_config(path).errors)


def test_eft_poi_and_disjoint_normfactors(tmp_path):
    path = edit(tmp_path, 'POI: "mu_H"', 'POI: "ctW"')
    path.write_text(path.read_text().replace("FitType: SPLUSB", "FitType: EFT") +
                    '\nSample: eft\n  Type: EFT\n  NtupleFiles: eft\n  EFTValue: ctW=1\n')
    assert verify_config(path).analysis_valid
    path = edit(tmp_path, 'NormFactor: "mu_yy_cat_central_lowptt"', 'NormFactor: "mu_yy_cat_2jet"')
    assert verify_config(path).analysis_valid  # Same name but disjoint attachments.


def test_scan_and_systematic_references(tmp_path):
    path = edit(tmp_path, "POIAsimov: 1", "POIAsimov: 1\n  doLHscan: mu_H\n  LHscanSteps: 2")
    path.write_text(path.read_text() + '\nSystematic: lumi\n  Type: OVERALL\n  Samples: nonexistent\n')
    codes = {i.code for i in verify_config(path).errors}
    assert {"numeric_range", "unknown_sample"} <= codes


def ntup_fixture(tmp_path):
    config = tmp_path / "tiny.config"
    config.write_text(f'''Job: tiny
  ReadFrom: NTUP
  NtuplePaths: "{tmp_path}"
  NtupleName: events
  SplitHistoFiles: TRUE
Region: sr
  Variable: x,2,0,2
  Selection: x>0
Sample: data
  Type: DATA
  NtupleFiles: input.root
''')
    with uproot.recreate(tmp_path / "input.root") as f:
        f.mktree("events", {"x": "float64"})
    return config


def test_optional_metadata_checks(tmp_path):
    config = ntup_fixture(tmp_path)
    with patch("uproot.open", side_effect=AssertionError("default must not open inputs")):
        assert verify_config(config).inputs_valid is None
    report = verify_config(config, check_inputs=True)
    assert report.valid and report.inputs_valid
    assert report.input_files_checked == 1
    config.write_text(config.read_text().replace("Selection: x>0", "Selection: missing>0"))
    report = verify_config(config, check_inputs=True)
    assert not report.inputs_valid
    assert any(i.code == "missing_branches" for i in report.input_issues)


def test_missing_pattern_and_tree(tmp_path):
    config = ntup_fixture(tmp_path)
    config.write_text(config.read_text().replace("input.root", "input.root,absent.root").replace("NtupleName: events", "NtupleName: missing"))
    report = verify_config(config, check_inputs=True)
    assert {i.code for i in report.input_issues} == {"missing_input", "missing_tree"}


def test_histogram_binning(tmp_path):
    config = tmp_path / "hist.config"
    config.write_text(f'''Job: hist
  ReadFrom: HIST
  HistoPath: "{tmp_path}"
Region: sr
  HistoName: h
Sample: a
  Type: DATA
  HistoFile: a
Sample: b
  Type: BACKGROUND
  HistoFile: b
''')
    for name, edges in [("a", [0., 1., 2.]), ("b", [0., 1., 3.])]:
        with uproot.recreate(tmp_path / f"{name}.root") as f:
            f["h"] = np.array([1., 2.]), np.array(edges)
    report = verify_config(config, check_inputs=True)
    assert report.analysis_valid and not report.coffea_compatible
    assert not report.inputs_valid
    assert any(i.code == "histogram_binning" for i in report.input_issues)


def test_combined_coffea_runner_actions():
    result = subprocess.run([sys.executable, str(ROOT / "trex_fitter/runner.py"),
                             str(HYY), "--backend", "coffea", "--actions", "nwsf", "--dry-run"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "Coffea action: n" in result.stdout
    assert "native container action string: wsf" in result.stdout
