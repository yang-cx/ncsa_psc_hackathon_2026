"""Compare Coffea output to a reference TRExFitter ``n`` output."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import uproot


@dataclass(frozen=True)
class Difference:
    file: str
    histogram: str
    max_content_absolute: float
    max_content_relative: float
    max_variance_absolute: float
    max_variance_relative: float
    class_equal: bool
    edges_equal: bool
    title_equal: bool
    x_axis_title_equal: bool
    y_axis_title_equal: bool
    within_tolerance: bool


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    return float(np.max(np.abs(left - right) / denominator, initial=0.0))


def compare_outputs(
    reference: Path,
    candidate: Path,
    *,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
) -> list[Difference]:
    reference_files = sorted((reference / "Histograms").glob("*_histos.root"))
    if not reference_files:
        raise ValueError(f"No split histogram ROOT files found under {reference}")
    differences: list[Difference] = []
    for reference_file in reference_files:
        candidate_file = candidate / "Histograms" / reference_file.name
        if not candidate_file.is_file():
            raise ValueError(f"Candidate is missing {candidate_file}")
        with uproot.open(reference_file) as expected, uproot.open(candidate_file) as actual:
            expected_keys = sorted(
                key.split(";")[0]
                for key, class_name in expected.classnames(recursive=True).items()
                if class_name.startswith("TH1")
            )
            actual_keys = sorted(
                key.split(";")[0]
                for key, class_name in actual.classnames(recursive=True).items()
                if class_name.startswith("TH1")
            )
            if expected_keys != actual_keys:
                raise ValueError(
                    f"Histogram key mismatch in {reference_file.name}: "
                    f"reference={expected_keys}, candidate={actual_keys}"
                )
            for key in expected_keys:
                expected_hist = expected[key]
                actual_hist = actual[key]
                expected_values = expected_hist.values(flow=True)
                actual_values = actual_hist.values(flow=True)
                expected_variances = expected_hist.variances(flow=True)
                actual_variances = actual_hist.variances(flow=True)
                edges_equal = np.array_equal(
                    expected_hist.axis().edges(), actual_hist.axis().edges()
                )
                title_equal = expected_hist.title == actual_hist.title
                x_axis_title_equal = (
                    expected_hist.axis().member("fTitle")
                    == actual_hist.axis().member("fTitle")
                )
                y_axis_title_equal = (
                    expected_hist.member("fYaxis").member("fTitle")
                    == actual_hist.member("fYaxis").member("fTitle")
                )
                class_equal = expected_hist.classname == actual_hist.classname
                values_close = np.allclose(
                    expected_values, actual_values, rtol=rtol, atol=atol, equal_nan=True
                )
                variances_close = np.allclose(
                    expected_variances,
                    actual_variances,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
                differences.append(
                    Difference(
                        file=reference_file.name,
                        histogram=key,
                        max_content_absolute=float(
                            np.max(np.abs(expected_values - actual_values), initial=0.0)
                        ),
                        max_content_relative=_relative(expected_values, actual_values),
                        max_variance_absolute=float(
                            np.max(
                                np.abs(expected_variances - actual_variances), initial=0.0
                            )
                        ),
                        max_variance_relative=_relative(
                            expected_variances, actual_variances
                        ),
                        class_equal=class_equal,
                        edges_equal=edges_equal,
                        title_equal=title_equal,
                        x_axis_title_equal=x_axis_title_equal,
                        y_axis_title_equal=y_axis_title_equal,
                        within_tolerance=(
                            class_equal
                            and edges_equal
                            and title_equal
                            and x_axis_title_equal
                            and y_axis_title_equal
                            and values_close
                            and variances_close
                        ),
                    )
                )
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    differences = compare_outputs(
        args.reference, args.candidate, rtol=args.rtol, atol=args.atol
    )
    failed = [difference for difference in differences if not difference.within_tolerance]
    summary = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "histograms": len(differences),
        "passed": len(differences) - len(failed),
        "failed": len(failed),
        "rtol": args.rtol,
        "atol": args.atol,
        "maximum_content_relative": max(
            (item.max_content_relative for item in differences), default=0.0
        ),
        "maximum_variance_relative": max(
            (item.max_variance_relative for item in differences), default=0.0
        ),
        "failures": [asdict(item) for item in failed],
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
