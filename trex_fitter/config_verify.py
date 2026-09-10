"""Fast, host-side validation of a basic TRExFitter analysis config.

The verifier covers the complete basic analysis model used by ``hyy.config``:
Job, Fit, Region, Sample, and NormFactor blocks, including cross-references.
It does not open ROOT files, start a container, or decide whether an optional
histogramming backend supports the config.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .coffea_backend.config import ConfigError, _Block, _parse_blocks, _unquote, split_top_level


class ConfigIssue(BaseModel):
    code: str
    message: str
    block: str | None = None
    name: str | None = None
    setting: str | None = None
    line: int | None = None


class ConfigReport(BaseModel):
    verifier: str = "trex-basic-analysis-v1.10"
    config: str
    valid: bool
    blocks: dict[str, int]
    errors: list[ConfigIssue]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ConfigError("; ".join(issue.message for issue in self.errors))


class _Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_from: Literal["NTUP", "HIST"]
    poi: str
    luminosity: float = Field(gt=0)
    split_histo_files: bool | None = None


class _Fit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_type: Literal["BONLY", "SPLUSB"]
    fit_region: Literal["CRONLY", "CRSR"]
    poi_asimov: float | None = None


class _Region(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_type: Literal["SIGNAL", "CONTROL", "VALIDATION"]
    expression: str
    bins: int = Field(gt=0)
    minimum: float
    maximum: float
    selection: str

    @model_validator(mode="after")
    def increasing_axis(self):
        if self.maximum <= self.minimum:
            raise ValueError("Variable maximum must be greater than minimum")
        return self


class _Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_type: Literal["DATA", "BACKGROUND", "SIGNAL"]
    files: list[str] = Field(min_length=1)


class _NormFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[str] = Field(min_length=1)
    regions: list[str] = Field(default_factory=list)
    nominal: float
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def valid_range(self):
        if self.maximum <= self.minimum:
            raise ValueError("Max must be greater than Min")
        if not self.minimum <= self.nominal <= self.maximum:
            raise ValueError("Nominal must lie between Min and Max")
        return self


# Source-grounded basic profile for TRExFitter v1.10.0. These are analysis
# settings, not Coffea capabilities. Extend this profile deliberately when a
# new atomic config operation is added to the project.
_SETTINGS = {
    "Job": {
        "Label", "CmeLabel", "POI", "ReadFrom", "NtuplePaths", "NtupleName",
        "HistoPath", "LumiLabel", "Lumi", "DebugLevel", "PlotOptions",
        "SplitHistoFiles", "SystControlPlots", "SystCategoryTables",
        "CorrelationThreshold", "MCstatThreshold",
    },
    "Fit": {"FitType", "FitRegion", "POIAsimov", "UseMinos", "FitBlind", "doLHscan"},
    "Region": {"Type", "Variable", "VariableTitle", "Selection", "Label", "ShortLabel", "HistoName"},
    "Sample": {
        "Type", "Title", "FillColor", "LineColor", "NtuplePathSuff",
        "NtupleFiles", "NtupleFile", "MCweight", "Selection", "UseMCstat",
        "HistoFile", "HistoName", "NormFactor", "Regions",
    },
    "NormFactor": {"Samples", "Regions", "Title", "Nominal", "Min", "Max"},
}


def _issue(block: _Block, code: str, message: str, setting: str | None = None) -> ConfigIssue:
    return ConfigIssue(
        code=code,
        message=message,
        block=block.kind,
        name=block.name,
        setting=setting,
        line=block.line,
    )


def _boolean(raw: str) -> bool:
    value = _unquote(raw).upper()
    if value not in {"TRUE", "FALSE"}:
        raise ValueError(f"expected TRUE or FALSE, got {raw!r}")
    return value == "TRUE"


def _model_errors(block: _Block, error: ValidationError) -> list[ConfigIssue]:
    return [
        _issue(
            block,
            "invalid_value",
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}",
        )
        for detail in error.errors(include_url=False)
    ]


def verify_config(path: Path | str) -> ConfigReport:
    """Validate the project's complete basic analysis profile in milliseconds."""
    path = Path(path).resolve()
    try:
        blocks = _parse_blocks(path)
    except (ConfigError, OSError) as error:
        return ConfigReport(
            config=str(path), valid=False, blocks={},
            errors=[ConfigIssue(code="parse_error", message=str(error))],
        )

    counts = dict(Counter(block.kind for block in blocks))
    errors: list[ConfigIssue] = []
    by_kind = {kind: [block for block in blocks if block.kind == kind] for kind in _SETTINGS}

    if len(by_kind["Job"]) != 1:
        errors.append(ConfigIssue(code="job_count", message=f"expected exactly one Job block, found {len(by_kind['Job'])}"))
    if len(by_kind["Fit"]) > 1:
        errors.append(ConfigIssue(code="fit_count", message=f"expected at most one Fit block, found {len(by_kind['Fit'])}"))
    for kind in ("Region", "Sample"):
        if not by_kind[kind]:
            errors.append(ConfigIssue(code=f"missing_{kind.lower()}", message=f"no {kind} blocks found"))

    for block in blocks:
        if block.kind not in _SETTINGS:
            errors.append(_issue(block, "unsupported_block", f"{block.kind} is outside the project's basic analysis profile"))
            continue
        for setting in block.values:
            if setting not in _SETTINGS[block.kind]:
                errors.append(_issue(block, "unknown_setting", f"{setting} is not valid in a basic {block.kind} block", setting))

    job_mode: str | None = None
    if len(by_kind["Job"]) == 1:
        block = by_kind["Job"][0]
        try:
            job_mode = _unquote(block.values.get("ReadFrom", "")).upper()
            _Job.model_validate({
                "read_from": job_mode,
                "poi": _unquote(block.values.get("POI", "")),
                "luminosity": _unquote(block.values.get("Lumi", "1")),
                "split_histo_files": _boolean(block.values["SplitHistoFiles"]) if "SplitHistoFiles" in block.values else None,
            })
            required = "NtuplePaths" if job_mode == "NTUP" else "HistoPath"
            if not split_top_level(block.values.get(required, "")):
                errors.append(_issue(block, "missing_setting", f"{required} is required for ReadFrom: {job_mode}", required))
        except ValidationError as error:
            errors.extend(_model_errors(block, error))
        except ValueError as error:
            errors.append(_issue(block, "invalid_value", str(error)))

    for block in by_kind["Fit"]:
        try:
            _Fit.model_validate({
                "fit_type": _unquote(block.values.get("FitType", "")).upper(),
                "fit_region": _unquote(block.values.get("FitRegion", "")).upper(),
                "poi_asimov": _unquote(block.values["POIAsimov"]) if "POIAsimov" in block.values else None,
            })
        except ValidationError as error:
            errors.extend(_model_errors(block, error))

    for block in by_kind["Region"]:
        if job_mode == "NTUP":
            variable = split_top_level(block.values.get("Variable", ""))
            if len(variable) != 4:
                errors.append(_issue(block, "invalid_variable", "Variable requires expression, bins, minimum, maximum", "Variable"))
                continue
            try:
                _Region.model_validate({
                    "region_type": _unquote(block.values.get("Type", "SIGNAL")).upper(),
                    "expression": variable[0], "bins": variable[1],
                    "minimum": variable[2], "maximum": variable[3],
                    "selection": _unquote(block.values.get("Selection", "")),
                })
            except ValidationError as error:
                errors.extend(_model_errors(block, error))
        elif job_mode == "HIST" and not _unquote(block.values.get("HistoName", "")):
            errors.append(_issue(block, "missing_setting", "HistoName is required for ReadFrom: HIST", "HistoName"))

    for block in by_kind["Sample"]:
        file_setting = "NtupleFiles" if "NtupleFiles" in block.values else "NtupleFile"
        if job_mode == "HIST":
            file_setting = "HistoFile"
        try:
            _Sample.model_validate({
                "sample_type": _unquote(block.values.get("Type", "")).upper(),
                "files": split_top_level(block.values.get(file_setting, "")),
            })
        except ValidationError as error:
            errors.extend(_model_errors(block, error))

    sample_names = {block.name for block in by_kind["Sample"]}
    region_names = {block.name for block in by_kind["Region"]}
    for block in by_kind["NormFactor"]:
        try:
            model = _NormFactor.model_validate({
                "samples": split_top_level(block.values.get("Samples", "")),
                "regions": split_top_level(block.values.get("Regions", "")),
                "nominal": _unquote(block.values.get("Nominal", "")),
                "minimum": _unquote(block.values.get("Min", "")),
                "maximum": _unquote(block.values.get("Max", "")),
            })
            missing_samples = sorted(set(model.samples) - sample_names)
            missing_regions = sorted(set(model.regions) - region_names)
            if missing_samples:
                errors.append(_issue(block, "unknown_sample", f"unknown Samples reference(s): {', '.join(missing_samples)}", "Samples"))
            if missing_regions:
                errors.append(_issue(block, "unknown_region", f"unknown Regions reference(s): {', '.join(missing_regions)}", "Regions"))
        except ValidationError as error:
            errors.extend(_model_errors(block, error))

    for kind, selected in by_kind.items():
        names = [block.name for block in selected]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            errors.append(ConfigIssue(code="duplicate_name", message=f"duplicate {kind} names: {', '.join(duplicates)}", block=kind))

    if len(by_kind["Job"]) == 1:
        poi = _unquote(by_kind["Job"][0].values.get("POI", ""))
        declared = {block.name for block in by_kind["NormFactor"]}
        if poi and poi not in declared:
            errors.append(_issue(by_kind["Job"][0], "unknown_poi", f"POI {poi!r} does not name a NormFactor block", "POI"))

    return ConfigReport(config=str(path), valid=not errors, blocks=counts, errors=errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path", help="also write the full machine-readable report")
    args = parser.parse_args()
    report = verify_config(args.config)
    status = "VALID" if report.valid else "INVALID"
    counts = ", ".join(f"{kind}={count}" for kind, count in sorted(report.blocks.items()))
    print(f"{status}: {args.config} ({counts})")
    for issue in report.errors:
        where = f"line {issue.line}: " if issue.line else ""
        print(f"  ERROR [{issue.code}] {where}{issue.message}")
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
