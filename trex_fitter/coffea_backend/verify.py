"""Check whether the Coffea backend can replace TRExFitter action ``n``.

This is a backend capability check, not the analysis config verifier. Blocks
and settings consumed only by later, native TRExFitter stages are accepted
silently because the Coffea backend does not replace those stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import ConfigError, _Block, _parse_blocks, _unquote, split_top_level
from .expressions import Expression


class VerificationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    block: str | None = None
    name: str | None = None
    setting: str | None = None
    line: int | None = None


class VerificationReport(BaseModel):
    verifier: str = "coffea-nominal-ntup-v1"
    config: str
    compatible: bool
    jobs: int
    regions: int
    samples: int
    issues: list[VerificationIssue]

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            rendered = "; ".join(
                f"{item.block or 'Config'} {item.name or ''}: {item.message}".strip()
                for item in errors
            )
            raise ConfigError("Coffea compatibility verification failed: " + rendered)


class _JobModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    read_from: Literal["NTUP"]
    ntuple_paths: list[str] = Field(min_length=1)
    ntuple_name: str = "nominal"
    luminosity: float = Field(default=1.0, gt=0)
    split_histo_files: bool


class _RegionModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    region_type: Literal["SIGNAL", "CONTROL", "VALIDATION"] = "SIGNAL"
    variable: str
    bins: int = Field(gt=0)
    minimum: float
    maximum: float
    selection: str

    @model_validator(mode="after")
    def increasing_axis(self):
        if self.maximum <= self.minimum:
            raise ValueError("Variable maximum must be greater than minimum")
        return self


class _SampleModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_type: Literal["DATA", "BACKGROUND", "SIGNAL"]
    ntuple_files: list[str] = Field(min_length=1)
    selection: str = "1"
    mc_weight: str | None = None

    @model_validator(mode="after")
    def weight_for_simulation(self):
        if self.sample_type != "DATA" and not self.mc_weight:
            raise ValueError("MCweight is required for BACKGROUND and SIGNAL samples")
        return self


_SUPPORTED_SETTINGS = {
    "Job": {
        "ReadFrom",
        "NtuplePaths",
        "NtupleName",
        "Lumi",
        "SplitHistoFiles",
    },
    "Region": {"Type", "Variable", "VariableTitle", "Selection"},
    "Sample": {
        "Type",
        "Title",
        "NtuplePathSuff",
        "NtupleFiles",
        "MCweight",
        "Selection",
    },
}

_DOWNSTREAM_SETTINGS = {
    "Job": {
        "Label",
        "CmeLabel",
        "POI",
        "LumiLabel",
        "DebugLevel",
        "PlotOptions",
        "SystControlPlots",
        "SystCategoryTables",
        "CorrelationThreshold",
        "MCstatThreshold",
    },
    "Region": {"Label", "ShortLabel"},
    "Sample": {"FillColor", "LineColor", "UseMCstat"},
}

_DOWNSTREAM_BLOCKS = {"Fit", "NormFactor", "ShapeFactor", "Limit", "Significance"}
_HISTOGRAM_AFFECTING_BLOCKS = {
    "Systematic",
    "Unfolding",
    "EFTConfig",
    "Morphing",
}


def _location(block: _Block, setting: str | None = None) -> dict[str, Any]:
    return {
        "block": block.kind,
        "name": block.name,
        "setting": setting,
        "line": block.line,
    }


def _pydantic_issues(block: _Block, error: ValidationError) -> list[VerificationIssue]:
    issues = []
    for detail in error.errors(include_url=False):
        location = ".".join(str(item) for item in detail["loc"])
        issues.append(
            VerificationIssue(
                severity="error",
                code="invalid_value",
                message=f"{location}: {detail['msg']}",
                **_location(block),
            )
        )
    return issues


def _boolean(raw: str) -> bool:
    value = _unquote(raw).upper()
    if value not in {"TRUE", "FALSE"}:
        raise ValueError(f"expected TRUE or FALSE, got {raw!r}")
    return value == "TRUE"


def _validate_job(block: _Block) -> _JobModel:
    return _JobModel.model_validate(
        {
            "read_from": _unquote(block.values.get("ReadFrom", "")).upper(),
            "ntuple_paths": split_top_level(block.values.get("NtuplePaths", "")),
            "ntuple_name": _unquote(block.values.get("NtupleName", "nominal")),
            "luminosity": _unquote(block.values.get("Lumi", "1")),
            "split_histo_files": _boolean(block.values.get("SplitHistoFiles", "FALSE")),
        }
    )


def _validate_region(block: _Block) -> _RegionModel:
    variable = split_top_level(block.values.get("Variable", ""))
    if len(variable) != 4:
        raise ValueError("Variable requires expression, bins, minimum, maximum")
    return _RegionModel.model_validate(
        {
            "region_type": _unquote(block.values.get("Type", "SIGNAL")).upper(),
            "variable": variable[0],
            "bins": variable[1],
            "minimum": variable[2],
            "maximum": variable[3],
            "selection": _unquote(block.values.get("Selection", "")),
        }
    )


def _validate_sample(block: _Block) -> _SampleModel:
    return _SampleModel.model_validate(
        {
            "sample_type": _unquote(block.values.get("Type", "")).upper(),
            "ntuple_files": split_top_level(block.values.get("NtupleFiles", "")),
            "selection": _unquote(block.values.get("Selection", "1")),
            "mc_weight": (
                _unquote(block.values["MCweight"])
                if "MCweight" in block.values
                else None
            ),
        }
    )


def verify_config(path: Path | str) -> VerificationReport:
    """Validate a config without opening inputs or invoking TRExFitter."""
    path = Path(path).resolve()
    issues: list[VerificationIssue] = []
    try:
        blocks = _parse_blocks(path)
    except (ConfigError, OSError) as error:
        return VerificationReport(
            config=str(path),
            compatible=False,
            jobs=0,
            regions=0,
            samples=0,
            issues=[
                VerificationIssue(
                    severity="error", code="parse_error", message=str(error)
                )
            ],
        )

    jobs = [block for block in blocks if block.kind == "Job"]
    regions = [block for block in blocks if block.kind == "Region"]
    samples = [block for block in blocks if block.kind == "Sample"]
    if len(jobs) != 1:
        issues.append(
            VerificationIssue(
                severity="error",
                code="job_count",
                message=f"expected exactly one Job block, found {len(jobs)}",
            )
        )
    if not regions:
        issues.append(
            VerificationIssue(
                severity="error", code="missing_region", message="no Region blocks found"
            )
        )
    if not samples:
        issues.append(
            VerificationIssue(
                severity="error", code="missing_sample", message="no Sample blocks found"
            )
        )

    for block in blocks:
        if block.kind in _DOWNSTREAM_BLOCKS:
            continue
        if block.kind in _HISTOGRAM_AFFECTING_BLOCKS:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="unsupported_block",
                    message=(
                        "histogram-affecting block is not implemented by the "
                        "nominal backend"
                    ),
                    **_location(block),
                )
            )
            continue
        if block.kind not in _SUPPORTED_SETTINGS:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="unknown_block",
                    message="block is outside the verified nominal-NTUP subset",
                    **_location(block),
                )
            )
            continue

        supported = _SUPPORTED_SETTINGS[block.kind]
        downstream = _DOWNSTREAM_SETTINGS[block.kind]
        for setting in block.values:
            if setting in supported:
                continue
            if setting in downstream:
                continue
            else:
                issues.append(
                    VerificationIssue(
                        severity="error",
                        code="unsupported_setting",
                        message="setting is not implemented by the Coffea backend",
                        **_location(block, setting),
                    )
                )

        try:
            if block.kind == "Job":
                model = _validate_job(block)
                if not model.split_histo_files:
                    issues.append(
                        VerificationIssue(
                            severity="error",
                            code="split_histograms_required",
                            message="SplitHistoFiles must be TRUE for the current writer",
                            **_location(block, "SplitHistoFiles"),
                        )
                    )
            elif block.kind == "Region":
                _validate_region(block)
                for setting in ("Variable", "Selection"):
                    raw = block.values.get(setting, "")
                    expression = (
                        split_top_level(raw)[0] if setting == "Variable" else _unquote(raw)
                    )
                    try:
                        Expression(expression)
                    except (SyntaxError, ValueError) as error:
                        issues.append(
                            VerificationIssue(
                                severity="error",
                                code="unsupported_expression",
                                message=(
                                    "expression cannot be evaluated safely: "
                                    f"{error}"
                                ),
                                **_location(block, setting),
                            )
                        )
            else:
                sample = _validate_sample(block)
                expression_settings = ["Selection"]
                if "MCweight" in block.values:
                    expression_settings.append("MCweight")
                for setting in expression_settings:
                    try:
                        Expression(_unquote(block.values.get(setting, "1")))
                    except (SyntaxError, ValueError) as error:
                        issues.append(
                            VerificationIssue(
                                severity="error",
                                code="unsupported_expression",
                                message=(
                                    "expression cannot be evaluated safely: "
                                    f"{error}"
                                ),
                                **_location(block, setting),
                            )
                        )
                if sample.sample_type == "DATA" and sample.mc_weight:
                    issues.append(
                        VerificationIssue(
                            severity="warning",
                            code="data_weight_ignored",
                            message=(
                                "TRExFitter uses unit weight for DATA; MCweight is "
                                "ignored"
                            ),
                            **_location(block, "MCweight"),
                        )
                    )
        except ValidationError as error:
            issues.extend(_pydantic_issues(block, error))
        except (TypeError, ValueError) as error:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="invalid_value",
                    message=str(error),
                    **_location(block),
                )
            )

    for kind, selected in (("Region", regions), ("Sample", samples)):
        names = [block.name for block in selected]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="duplicate_name",
                    message=f"duplicate {kind} names: {', '.join(duplicates)}",
                )
            )

    return VerificationReport(
        config=str(path),
        compatible=not any(issue.severity == "error" for issue in issues),
        jobs=len(jobs),
        regions=len(regions),
        samples=len(samples),
        issues=issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    report = verify_config(args.config)
    rendered = json.dumps(report.model_dump(), indent=2)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n")
    raise SystemExit(0 if report.compatible else 1)


if __name__ == "__main__":
    main()
