from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import awkward as ak
import hist
import numpy as np
import uproot

from trex_fitter.coffea_backend.backend import _stage_files
from trex_fitter.coffea_backend.config import parse_config, split_top_level
from trex_fitter.coffea_backend.expressions import Expression, boolean_mask
from trex_fitter.coffea_backend.verify import verify_config
from trex_fitter.coffea_backend.writer import _fold_flow, _make_histogram, _sanitize


REPOSITORY = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_hyy_config(self):
        config = parse_config(REPOSITORY / "data/configs/examples/hyy.config")
        self.assertEqual(config.name, "hyy")
        self.assertEqual(config.ntuple_name, "analysis")
        self.assertEqual(config.luminosity, 36100)
        self.assertEqual(len(config.regions), 6)
        self.assertEqual(len(config.samples), 7)
        self.assertEqual(config.regions[0].bins, 15)
        self.assertEqual(config.samples[0].name, "Data")
        self.assertTrue(config.samples[0].is_data)
        self.assertEqual(len(config.samples[1].file_patterns), 4)

    def test_stage_files_preserves_sample_mapping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.root"
            source.write_bytes(b"root-data")
            staged, total_bytes = _stage_files(
                {"sample": [str(source)]}, root / "stage"
            )
            destination = Path(staged["sample"][0])
            self.assertEqual(destination.read_bytes(), b"root-data")
            self.assertEqual(total_bytes, len(b"root-data"))

    def test_split_preserves_function_commas(self):
        self.assertEqual(
            split_top_level('"atan2(y,x)",15,100,160'),
            ["atan2(y,x)", "15", "100", "160"],
        )

    def test_hyy_is_compatible_with_static_verifier(self):
        report = verify_config(REPOSITORY / "data/configs/examples/hyy.config")
        self.assertTrue(report.compatible)
        self.assertEqual((report.jobs, report.regions, report.samples), (1, 6, 7))
        self.assertTrue(any(issue.code == "downstream_block" for issue in report.issues))

    def test_verifier_rejects_unsupported_histogram_setting(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsupported.config"
            path.write_text(
                'Job: "x"\n'
                '  ReadFrom: NTUP\n'
                '  NtuplePaths: "inputs"\n'
                '  SplitHistoFiles: TRUE\n'
                '  Selection: "event_clean"\n'
                'Region: "sr"\n'
                '  Variable: "x",10,0,1\n'
                '  Selection: "1"\n'
                'Sample: "data"\n'
                '  Type: DATA\n'
                '  NtupleFiles: "data"\n'
            )
            report = verify_config(path)
            self.assertFalse(report.compatible)
            self.assertTrue(any(
                issue.code == "unsupported_setting" and issue.setting == "Selection"
                for issue in report.issues
            ))

    def test_verifier_requires_mc_weight(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-weight.config"
            path.write_text(
                'Job: "x"\n'
                '  ReadFrom: NTUP\n'
                '  NtuplePaths: "inputs"\n'
                '  SplitHistoFiles: TRUE\n'
                'Region: "sr"\n'
                '  Variable: "x",10,0,1\n'
                '  Selection: "1"\n'
                'Sample: "background"\n'
                '  Type: BACKGROUND\n'
                '  NtupleFiles: "mc"\n'
            )
            report = verify_config(path)
            self.assertFalse(report.compatible)
            self.assertTrue(
                any("MCweight is required" in issue.message for issue in report.issues)
            )

    def test_verifier_rejects_expression_outside_safe_subset(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "expression.config"
            path.write_text(
                'Job: "x"\n'
                '  ReadFrom: NTUP\n'
                '  NtuplePaths: "inputs"\n'
                '  SplitHistoFiles: TRUE\n'
                'Region: "sr"\n'
                '  Variable: "unknown_root_function(x)",10,0,1\n'
                '  Selection: "1"\n'
                'Sample: "data"\n'
                '  Type: DATA\n'
                '  NtupleFiles: "data"\n'
            )
            report = verify_config(path)
            self.assertFalse(report.compatible)
            self.assertTrue(
                any(issue.code == "unsupported_expression" for issue in report.issues)
            )


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        self.events = ak.Array(
            {
                "n": [2, 1, 2],
                "pt": [[50.0, 40.0], [60.0], [20.0, 10.0]],
                "flag": [True, True, False],
                "phi": [[0.0, 1.0], [0.0], [1.0, 2.0]],
            }
        )

    def test_jagged_index_and_boolean_operators(self):
        expression = Expression("n>=2 && flag && pt[0]>40 && !(pt[1]<30)")
        np.testing.assert_array_equal(
            boolean_mask(expression(self.events)), [True, False, False]
        )

    def test_functions_and_names(self):
        expression = Expression("fabs(pt[0])*cos(phi[0]) + pow(pt[1],2)")
        self.assertEqual(expression.names, {"pt", "phi"})
        values = ak.to_list(expression(self.events))
        self.assertAlmostEqual(values[0], 1650.0)
        self.assertIsNone(values[1])

    def test_rejects_python_execution(self):
        with self.assertRaises(ValueError):
            Expression("__import__('os').system('id')")

    def test_scalar_config_boolean(self):
        np.testing.assert_array_equal(
            boolean_mask(Expression("TRUE")({}), size=3), [True, True, True]
        )


class WriterTests(unittest.TestCase):
    def test_flow_folding(self):
        source = hist.Hist(
            hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight()
        )
        source.fill([-1.0, 0.5, 1.5, 3.0], weight=[2.0, 3.0, 4.0, 5.0])
        values, variances = _fold_flow(source)
        np.testing.assert_allclose(values, [5.0, 9.0])
        np.testing.assert_allclose(variances, [13.0, 41.0])

    def test_mc_zero_bin_repair_preserves_integral(self):
        values = np.array([2.0, 0.0, 4.0])
        variances = np.array([4.0, 0.0, 16.0])
        repaired_values, repaired_variances = _sanitize(
            values, variances, repair_empty_bins=True
        )
        self.assertAlmostEqual(repaired_values.sum(), values.sum())
        self.assertGreater(repaired_values[1], 0)
        self.assertEqual(repaired_variances[1], 4.0)

    def test_signal_empty_bins_are_unchanged(self):
        values = np.array([1.0, 0.0])
        variances = np.array([1.0, 0.0])
        actual_values, actual_variances = _sanitize(
            values, variances, repair_empty_bins=False
        )
        np.testing.assert_array_equal(actual_values, values)
        np.testing.assert_array_equal(actual_variances, variances)

    def test_empty_bin_repair_preserves_existing_error(self):
        values = np.array([2.0, -1.0, 4.0])
        variances = np.array([4.0, 25.0, 16.0])
        _, repaired_variances = _sanitize(values, variances, repair_empty_bins=True)
        self.assertEqual(repaired_variances[1], 25.0)

    def test_empty_bin_repair_fallback_matches_trex(self):
        values = np.array([0.0, -1.0])
        variances = np.array([0.0, 0.0])
        _, repaired_variances = _sanitize(values, variances, repair_empty_bins=True)
        np.testing.assert_array_equal(repaired_variances, [1.0e-12, 1.0e-12])

    def test_root_titles_match_trex_contract(self):
        config = parse_config(REPOSITORY / "data/configs/examples/hyy.config")
        region = config.regions[0]
        histogram = _make_histogram(
            region,
            np.ones(region.bins),
            np.ones(region.bins),
            title="#gamma#gamma continuum",
            variable_title=region.variable_title,
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "titles.root"
            with uproot.recreate(path) as output:
                output["h"] = histogram
            with uproot.open(path) as source:
                self.assertEqual(source["h"].title, "#gamma#gamma continuum")
                self.assertEqual(
                    source["h"].axis().member("fTitle"), region.variable_title
                )


class SftSeedTests(unittest.TestCase):
    def test_seed_rows_have_chat_shape_and_unique_ids(self):
        path = REPOSITORY / "trex_fitter/sft/trexfitter_atomic_examples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 8)
        identifiers = []
        for row in rows:
            self.assertEqual(
                [message["role"] for message in row["messages"]],
                ["system", "user", "assistant"],
            )
            metadata = row["metadata"]
            identifiers.append(metadata["id"])
            self.assertIn(
                metadata["support"], {"verified", "unsupported", "downstream"}
            )
            self.assertTrue(metadata["source_refs"])
        self.assertEqual(len(identifiers), len(set(identifiers)))


if __name__ == "__main__":
    unittest.main()
