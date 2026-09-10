"""Parse the config-driven subset of TRExFitter used by the Coffea backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """A setting cannot be represented faithfully by the Coffea backend."""


@dataclass(frozen=True)
class RegionConfig:
    name: str
    region_type: str
    variable: str
    bins: int
    minimum: float
    maximum: float
    variable_title: str
    selection: str


@dataclass(frozen=True)
class SampleConfig:
    name: str
    sample_type: str
    title: str
    path_suffix: str
    file_patterns: tuple[str, ...]
    selection: str
    weight: str | None

    @property
    def is_data(self) -> bool:
        return self.sample_type.upper() == "DATA"


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    name: str
    read_from: str
    ntuple_paths: tuple[str, ...]
    ntuple_name: str
    luminosity: float
    split_histo_files: bool
    regions: tuple[RegionConfig, ...]
    samples: tuple[SampleConfig, ...]


@dataclass
class _Block:
    kind: str
    name: str
    values: dict[str, str]
    line: int


def _without_comment(line: str) -> str:
    quoted = False
    escaped = False
    output: list[str] = []
    for character in line:
        if character == '"' and not escaped:
            quoted = not quoted
        if character == "%" and not quoted:
            break
        output.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return "".join(output).rstrip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def split_top_level(value: str) -> list[str]:
    """Split a TREx comma list, preserving quoted strings and function calls."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for character in value:
        if character == '"' and not escaped:
            quoted = not quoted
        elif not quoted:
            if character in "([":
                depth += 1
            elif character in ")]":
                depth -= 1
            elif character == "," and depth == 0:
                parts.append(_unquote("".join(current)))
                current = []
                continue
        current.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    parts.append(_unquote("".join(current)))
    return [part.strip() for part in parts if part.strip()]


def _parse_blocks(path: Path) -> list[_Block]:
    blocks: list[_Block] = []
    current: _Block | None = None
    for line_number, source_line in enumerate(path.read_text().splitlines(), 1):
        line = _without_comment(source_line)
        if not line.strip():
            continue
        if ":" not in line:
            raise ConfigError(f"{path}:{line_number}: expected 'key: value'")
        key, value = line.split(":", 1)
        if not source_line[:1].isspace():
            current = _Block(key.strip(), _unquote(value), {}, line_number)
            blocks.append(current)
        elif current is None:
            raise ConfigError(f"{path}:{line_number}: setting appears before a block")
        else:
            if key.strip() in current.values:
                raise ConfigError(
                    f"{path}:{line_number}: duplicate setting {key.strip()!r} "
                    f"in {current.kind} {current.name!r}"
                )
            current.values[key.strip()] = value.strip()
    return blocks


def _required(block: _Block, key: str, path: Path) -> str:
    return _unquote(_required_raw(block, key, path))


def _required_raw(block: _Block, key: str, path: Path) -> str:
    try:
        return block.values[key]
    except KeyError as exc:
        raise ConfigError(
            f"{path}:{block.line}: {block.kind} {block.name!r} requires {key}"
        ) from exc


def _boolean(value: str, *, setting: str) -> bool:
    normalized = _unquote(value).upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    raise ConfigError(f"{setting} must be TRUE or FALSE, got {value!r}")


def parse_config(path: Path | str) -> AnalysisConfig:
    """Parse a native TREx config without translating it to another format."""
    path = Path(path).resolve()
    blocks = _parse_blocks(path)
    jobs = [block for block in blocks if block.kind == "Job"]
    if len(jobs) != 1:
        raise ConfigError(f"{path}: expected exactly one Job block, found {len(jobs)}")
    job = jobs[0]
    read_from = _required(job, "ReadFrom", path).upper()
    if read_from != "NTUP":
        raise ConfigError(f"Coffea backend requires Job ReadFrom: NTUP, got {read_from}")

    systematics = [block for block in blocks if block.kind == "Systematic"]
    if systematics:
        names = ", ".join(block.name for block in systematics)
        raise ConfigError(
            "Coffea backend does not yet implement Systematic blocks; unsupported: " + names
        )

    regions: list[RegionConfig] = []
    for block in (item for item in blocks if item.kind == "Region"):
        variable = split_top_level(_required(block, "Variable", path))
        if len(variable) != 4:
            raise ConfigError(
                f"{path}:{block.line}: Region {block.name!r} Variable must have "
                "expression, bins, minimum, maximum"
            )
        regions.append(
            RegionConfig(
                name=block.name,
                region_type=_unquote(block.values.get("Type", "SIGNAL")),
                variable=variable[0],
                bins=int(variable[1]),
                minimum=float(variable[2]),
                maximum=float(variable[3]),
                variable_title=_unquote(block.values.get("VariableTitle", variable[0])),
                selection=_required(block, "Selection", path),
            )
        )

    samples: list[SampleConfig] = []
    for block in (item for item in blocks if item.kind == "Sample"):
        sample_type = _required(block, "Type", path).upper()
        if sample_type not in {"DATA", "BACKGROUND", "SIGNAL"}:
            raise ConfigError(
                f"{path}:{block.line}: unsupported sample Type {sample_type!r}"
            )
        samples.append(
            SampleConfig(
                name=block.name,
                sample_type=sample_type,
                title=_unquote(block.values.get("Title", block.name)),
                path_suffix=_unquote(block.values.get("NtuplePathSuff", "")),
                file_patterns=tuple(
                    split_top_level(_required_raw(block, "NtupleFiles", path))
                ),
                selection=_unquote(block.values.get("Selection", "1")),
                weight=(
                    None
                    if sample_type == "DATA"
                    else _required(block, "MCweight", path)
                ),
            )
        )

    if not regions:
        raise ConfigError(f"{path}: no Region blocks found")
    if not samples:
        raise ConfigError(f"{path}: no Sample blocks found")

    return AnalysisConfig(
        path=path,
        name=job.name,
        read_from=read_from,
        ntuple_paths=tuple(split_top_level(_required_raw(job, "NtuplePaths", path))),
        ntuple_name=_unquote(job.values.get("NtupleName", "nominal")),
        luminosity=float(_unquote(job.values.get("Lumi", "1"))),
        split_histo_files=_boolean(
            job.values.get("SplitHistoFiles", "FALSE"), setting="SplitHistoFiles"
        ),
        regions=tuple(regions),
        samples=tuple(samples),
    )


def _host_path(path: str, project_dir: Path) -> Path:
    input_prefix = "/workdir/inputs"
    if path == input_prefix:
        return project_dir / "data" / "samples"
    if path.startswith(input_prefix + "/"):
        return project_dir / "data" / "samples" / path[len(input_prefix) + 1 :]
    container_prefix = "/workdir"
    if path == container_prefix:
        return project_dir
    if path.startswith(container_prefix + "/"):
        return project_dir / path[len(container_prefix) + 1 :]
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate


def resolve_sample_files(
    config: AnalysisConfig, sample: SampleConfig, project_dir: Path
) -> list[Path]:
    """Resolve a sample's path suffix and ROOT filename patterns on the host."""
    matches: set[Path] = set()
    searched: list[str] = []
    for configured_path in config.ntuple_paths:
        directory = _host_path(configured_path, project_dir) / sample.path_suffix
        for pattern in sample.file_patterns:
            glob_pattern = pattern if pattern.endswith(".root") else pattern + "*.root"
            searched.append(str(directory / glob_pattern))
            matches.update(path.resolve() for path in directory.glob(glob_pattern))
    if not matches:
        raise ConfigError(
            f"No input files found for sample {sample.name!r}; searched: "
            + ", ".join(searched)
        )
    return sorted(matches)


def resolve_all_files(
    config: AnalysisConfig, project_dir: Path
) -> dict[str, list[str]]:
    return {
        sample.name: [
            str(path) for path in resolve_sample_files(config, sample, project_dir)
        ]
        for sample in config.samples
    }
