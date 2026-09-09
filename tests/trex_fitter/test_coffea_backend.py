from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import awkward as ak
import hist
import numpy as np

from trex_fitter.coffea_backend.backend import _stage_files
from trex_fitter.coffea_backend.config import parse_config, split_top_level
from trex_fitter.coffea_backend.expressions import Expression, boolean_mask
from trex_fitter.coffea_backend.writer import _fold_flow, _sanitize


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


if __name__ == "__main__":
    unittest.main()
