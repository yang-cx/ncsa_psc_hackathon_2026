"""Write Coffea histograms with the hierarchy expected by TRExFitter."""

from __future__ import annotations

from pathlib import Path

import hist
import numpy as np
import uproot
from uproot.writing import identify

from .config import AnalysisConfig, RegionConfig


def _fold_flow(source: hist.Hist) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(source.values(flow=True), dtype=np.float64)
    variances = np.asarray(source.variances(flow=True), dtype=np.float64)
    core_values = values[1:-1].copy()
    core_variances = variances[1:-1].copy()
    core_values[0] += values[0]
    core_values[-1] += values[-1]
    core_variances[0] += variances[0]
    core_variances[-1] += variances[-1]
    return core_values, core_variances


def _sanitize(
    values: np.ndarray, variances: np.ndarray, *, repair_empty_bins: bool
) -> tuple[np.ndarray, np.ndarray]:
    nominal_values = values.copy()
    nominal_variances = variances.copy()
    if not repair_empty_bins:
        return nominal_values, nominal_variances
    bad = nominal_values <= 0
    if not np.any(bad):
        return nominal_values, nominal_variances
    # SampleHist::FixEmptyBins in TRExFitter v1.10.0 preserves an existing
    # error in a non-positive bin. For an errorless bin it borrows the smallest
    # error from a positive-content bin, falling back to 1e-6.
    donors = (nominal_values > 0) & (nominal_variances > 0)
    positive_errors = np.sqrt(nominal_variances[donors])
    replacement_error = (
        float(positive_errors.min()) if len(positive_errors) else 1.0e-6
    )
    original_integral = float(nominal_values.sum())
    nominal_values[bad] = 1.0e-6
    missing_error = bad & (nominal_variances <= 0)
    nominal_variances[missing_error] = replacement_error**2
    modified_integral = float(nominal_values.sum())
    if original_integral > 0 and modified_integral > 0:
        nominal_values *= original_integral / modified_integral
    return nominal_values, nominal_variances


def _make_histogram(
    region: RegionConfig,
    values: np.ndarray,
    variances: np.ndarray,
    *,
    title: str,
    variable_title: str,
) -> object:
    # Construct the writable ROOT object explicitly. hist.Hist deliberately
    # substitutes an empty axis label with its Python axis name, whereas the
    # TREx `_orig` objects have an exactly empty TAxis::fTitle.
    data = np.zeros(region.bins + 2, dtype=np.float64)
    sumw2 = np.zeros(region.bins + 2, dtype=np.float64)
    data[1:-1] = values
    sumw2[1:-1] = variances
    centers = np.linspace(
        region.minimum + (region.maximum - region.minimum) / (2 * region.bins),
        region.maximum - (region.maximum - region.minimum) / (2 * region.bins),
        region.bins,
    )
    return identify.to_TH1x(
        fName=None,
        fTitle=title,
        data=data,
        fEntries=float(values.sum()),
        fTsumw=float(values.sum()),
        fTsumw2=float(variances.sum()),
        fTsumwx=float(np.dot(values, centers)),
        fTsumwx2=float(np.dot(values, centers**2)),
        fSumw2=sumw2,
        fXaxis=identify.to_TAxis(
            fName="xaxis",
            fTitle=variable_title,
            fNbins=region.bins,
            fXmin=region.minimum,
            fXmax=region.maximum,
        ),
    )


def write_histograms(
    config: AnalysisConfig,
    result: dict[str, dict[str, hist.Hist]],
    output_base: Path,
) -> Path:
    """Write split histogram files and return the analysis output directory."""
    analysis_dir = output_base / config.name
    histogram_dir = analysis_dir / "Histograms"
    binning_dir = histogram_dir / "Binnings"
    binning_dir.mkdir(parents=True, exist_ok=True)

    for region in config.regions:
        root_path = histogram_dir / f"{config.name}_{region.name}_histos.root"
        with uproot.recreate(root_path) as root_file:
            for sample in config.samples:
                try:
                    source = result[sample.name][region.name]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Coffea result is missing {sample.name}/{region.name}"
                    ) from exc
                original_values, original_variances = _fold_flow(source)
                nominal_values, nominal_variances = _sanitize(
                    original_values,
                    original_variances,
                    repair_empty_bins=sample.sample_type.upper() == "BACKGROUND",
                )
                prefix = f"{region.name}/{sample.name}/nominal/{region.name}_{sample.name}"
                root_file[prefix + "_orig"] = _make_histogram(
                    region,
                    original_values,
                    original_variances,
                    title="h",
                    variable_title="",
                )
                root_file[prefix] = _make_histogram(
                    region,
                    nominal_values,
                    nominal_variances,
                    title=sample.title,
                    variable_title=region.variable_title,
                )
                root_file[prefix + "_regBin"] = _make_histogram(
                    region,
                    nominal_values,
                    nominal_variances,
                    title=sample.title,
                    variable_title=region.variable_title,
                )
        (binning_dir / f"{region.name}.txt").write_text(f"{region.bins}\n")
    return analysis_dir
