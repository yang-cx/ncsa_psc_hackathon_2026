"""Fast, host-side validation of a basic TRExFitter analysis config.

The verifier covers the complete basic analysis model used by ``hyy.config``:
Job, Fit, Region, Sample, and NormFactor blocks, including cross-references.
It checks native setting types against pinned v1.10.0 schemas and reports
Coffea compatibility without opening ROOT files or starting a container.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config_format import ConfigError, _Block, _parse_blocks, _unquote, split_top_level
from .schema import load_schema, matches_schema


class ConfigIssue(BaseModel):
    code: str
    message: str
    block: str | None = None
    name: str | None = None
    setting: str | None = None
    line: int | None = None
    source_ref: str | None = None


class _AnalysisReport(BaseModel):
    config: str
    valid: bool
    blocks: dict[str, int]
    errors: list[ConfigIssue]


class VerificationReport(BaseModel):
    """One verdict with granular analysis and Coffea compatibility results."""

    verifier: str = "trex-basic-analysis-and-coffea-v1.10"
    config: str
    valid: bool
    analysis_valid: bool
    coffea_compatible: bool
    blocks: dict[str, int]
    errors: list[ConfigIssue]
    coffea_issues: list[dict[str, Any]]
    inputs_valid: bool | None = None
    input_issues: list[ConfigIssue] = Field(default_factory=list)
    input_files_checked: int = 0

    def raise_for_errors(self) -> None:
        messages = [issue.message for issue in self.errors]
        messages.extend(str(issue["message"]) for issue in self.coffea_issues if issue["severity"] == "error")
        messages.extend(issue.message for issue in self.input_issues)
        if messages:
            raise ConfigError("; ".join(messages))


class _Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_from: Literal["NTUP", "HIST"]
    poi: str
    luminosity: float = Field(gt=0)
    split_histo_files: bool | None = None


class _Fit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_type: Literal["BONLY", "SPLUSB", "UNFOLDING", "EFT"]
    fit_region: Literal["CRONLY", "CRSR"]
    poi_asimov: str | None = None


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

    sample_type: Literal["DATA", "BACKGROUND", "SIGNAL", "GHOST", "EFT"]
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


_SEMANTIC_BLOCKS = ("Job", "Fit", "Region", "Sample", "NormFactor")


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


def _verify_analysis_config(path: Path | str, actions="nwfs") -> _AnalysisReport:
    """Validate the project's complete basic analysis profile."""
    path = Path(path).resolve()
    try:
        blocks = _parse_blocks(path)
    except (ConfigError, OSError) as error:
        return _AnalysisReport(
            config=str(path), valid=False, blocks={},
            errors=[ConfigIssue(code="parse_error", message=str(error))],
        )

    counts = dict(Counter(block.kind for block in blocks))
    errors: list[ConfigIssue] = []
    schema = load_schema(any(block.kind == "MultiFit" for block in blocks))
    by_kind = {kind: [block for block in blocks if block.kind == kind] for kind in _SEMANTIC_BLOCKS}

    if len(by_kind["Job"]) != 1:
        errors.append(ConfigIssue(code="job_count", message=f"expected exactly one Job block, found {len(by_kind['Job'])}"))
    if len(by_kind["Fit"]) > 1:
        errors.append(ConfigIssue(code="fit_count", message=f"expected at most one Fit block, found {len(by_kind['Fit'])}"))
    for kind in ("Region", "Sample"):
        if not by_kind[kind]:
            errors.append(ConfigIssue(code=f"missing_{kind.lower()}", message=f"no {kind} blocks found"))

    for block in blocks:
        if block.kind not in schema:
            errors.append(_issue(block, "unsupported_block", f"{block.kind} is absent from the TRExFitter v1.10.0 schema"))
            continue
        for setting, raw in block.values.items():
            specification = schema[block.kind].get(setting)
            if specification is None:
                errors.append(_issue(block, "unknown_setting", f"{setting} is not in the v1.10.0 {block.kind} schema", setting))
            elif not matches_schema(raw, specification):
                errors.append(_issue(block, "invalid_value", f"{setting}: expected {specification}, got {raw!r}", setting))

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
                "fit_type": _unquote(block.values.get("FitType", "SPLUSB")).upper(),
                "fit_region": _unquote(block.values.get("FitRegion", "CRSR")).upper(),
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

    for block in by_kind["NormFactor"]:
        try:
            model = _NormFactor.model_validate({
                "samples": split_top_level(block.values.get("Samples", "all")),
                "regions": split_top_level(block.values.get("Regions", "")),
                "nominal": _unquote(block.values.get("Nominal", "1")),
                "minimum": _unquote(block.values.get("Min", "0")),
                "maximum": _unquote(block.values.get("Max", "10")),
            })
        except ValidationError as error:
            errors.extend(_model_errors(block, error))

    for kind, selected in by_kind.items():
        if kind == "NormFactor":
            continue  # Check overlapping attachments rather than names alone.
        names = [block.name for block in selected]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            errors.append(ConfigIssue(code="duplicate_name", message=f"duplicate {kind} names: {', '.join(duplicates)}", block=kind))

    from .semantics import check_semantics
    errors.extend(ConfigIssue(**item) for item in check_semantics(blocks, actions))

    return _AnalysisReport(config=str(path), valid=not errors, blocks=counts, errors=errors)


def verify_config(path: Path | str, *, actions="nwfs", check_inputs=False, project_dir=None) -> VerificationReport:
    """Run both basic analysis validation and Coffea compatibility checking."""
    analysis = _verify_analysis_config(path, actions)
    from .coffea_backend.verify import verify_config as _verify_coffea_config

    coffea = _verify_coffea_config(path)
    report = VerificationReport(
        config=analysis.config,
        valid=analysis.valid and coffea.compatible,
        analysis_valid=analysis.valid,
        coffea_compatible=coffea.compatible,
        blocks=analysis.blocks,
        errors=analysis.errors,
        coffea_issues=[issue.model_dump() for issue in coffea.issues],
    )
    if check_inputs:
        from .input_checks import inspect_inputs
        findings, count = inspect_inputs(Path(path), Path(project_dir) if project_dir else Path(__file__).resolve().parents[1])
        report.input_issues = [ConfigIssue(**item) for item in findings]
        report.inputs_valid = not findings
        report.input_files_checked = count
        report.valid = report.valid and report.inputs_valid
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--actions", nargs="+", default=["n", "w", "f", "s"], help="Planned actions for conditional static checks (default: n w f s)")
    parser.add_argument("--check-inputs", action="store_true", help="Inspect ROOT metadata without reading event arrays; requires uproot")
    parser.add_argument("--project-dir", type=Path, help="Host root corresponding to /workdir (default: repository root)")
    parser.add_argument("--json", type=Path, dest="json_path", help="also write the full machine-readable report")
    args = parser.parse_args()
    report = verify_config(args.config, actions=args.actions, check_inputs=args.check_inputs, project_dir=args.project_dir)
    status = "VALID" if report.analysis_valid else "INVALID"
    counts = ", ".join(f"{kind}={count}" for kind, count in sorted(report.blocks.items()))
    print(f"{status}: {args.config} ({counts})")
    for issue in report.errors:
        where = f"line {issue.line}: " if issue.line else ""
        print(f"  ERROR [{issue.code}] {where}{issue.message}")
    coffea_status = "COMPATIBLE" if report.coffea_compatible else "INCOMPATIBLE"
    print(f"{coffea_status}: Coffea action n ({len(report.coffea_issues)} issue(s))")
    for issue in report.coffea_issues:
        where = f"line {issue['line']}: " if issue["line"] else ""
        print(f"  {issue['severity'].upper()} [{issue['code']}] {where}{issue['message']}")
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    if report.inputs_valid is not None:
        print(f"INPUTS {'VALID' if report.inputs_valid else 'INVALID'}: {report.input_files_checked} file(s) checked")
        for issue in report.input_issues:
            print(f"  ERROR [{issue.code}] {issue.message}")
    raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
