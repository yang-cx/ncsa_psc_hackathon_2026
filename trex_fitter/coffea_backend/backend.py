"""Orchestrate Coffea execution for a native TREx NTUP config."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import coffea
from coffea import processor
from coffea.nanoevents import BaseSchema

from ..config_verify import verify_config
from .config import parse_config, resolve_all_files
from .processor import TrexNtupleProcessor
from .writer import write_histograms


@dataclass(frozen=True)
class RunSummary:
    backend: str
    schema: str
    coffea_version: str
    config: str
    output: str
    samples: int
    files: int
    regions: int
    workers: int
    chunksize: int
    maxchunks: int | None
    stage_dir: str | None
    staged_bytes: int
    stage_seconds: float
    wall_seconds: float


def _schema(name: str):
    if name == "base":
        return BaseSchema
    if name == "atlas":
        try:
            from atlas_schema.schema import NtupleSchema
        except ImportError as exc:
            raise RuntimeError(
                "--coffea-schema atlas requires atlas-schema; run "
                "uv sync --extra coffea --extra atlas-schema"
            ) from exc
        return NtupleSchema
    raise ValueError(f"Unknown Coffea schema {name!r}")


def _stage_files(
    fileset: dict[str, list[str]], stage_dir: Path
) -> tuple[dict[str, list[str]], int]:
    """Copy ROOT inputs to fast local storage, reusing complete staged files."""
    staged: dict[str, list[str]] = {}
    total_bytes = 0
    for sample, filenames in fileset.items():
        sample_dir = stage_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        staged[sample] = []
        for filename in filenames:
            source = Path(filename)
            destination = sample_dir / source.name
            source_size = source.stat().st_size
            total_bytes += source_size
            if not destination.is_file() or destination.stat().st_size != source_size:
                partial = destination.with_suffix(destination.suffix + ".partial")
                shutil.copyfile(source, partial)
                partial.replace(destination)
            staged[sample].append(str(destination))
    return staged, total_bytes


def run_histogramming(
    config_path: Path,
    *,
    project_dir: Path,
    output_base: Path,
    workers: int = 1,
    chunksize: int = 250_000,
    maxchunks: int | None = None,
    schema: str = "base",
    stage_dir: Path | None = None,
) -> RunSummary:
    verify_config(config_path, actions="n").raise_for_errors()
    config = parse_config(config_path)
    if not config.split_histo_files:
        raise ValueError(
            "The first Coffea milestone requires Job SplitHistoFiles: TRUE"
        )
    fileset = resolve_all_files(config, project_dir.resolve())
    started = time.perf_counter()
    stage_started = time.perf_counter()
    staged_bytes = 0
    if stage_dir is not None:
        stage_dir = stage_dir.resolve()
        fileset, staged_bytes = _stage_files(fileset, stage_dir)
    stage_seconds = time.perf_counter() - stage_started
    executor = (
        processor.IterativeExecutor(status=True)
        if workers == 1
        else processor.FuturesExecutor(status=True, workers=workers)
    )
    runner = processor.Runner(
        executor=executor,
        chunksize=chunksize,
        maxchunks=maxchunks,
        schema=_schema(schema),
    )
    result = runner(
        fileset,
        treename=config.ntuple_name,
        processor_instance=TrexNtupleProcessor(config),
    )
    output_dir = write_histograms(config, result, output_base.resolve())
    elapsed = time.perf_counter() - started
    summary = RunSummary(
        backend="coffea",
        schema=schema,
        coffea_version=coffea.__version__,
        config=str(config.path),
        output=str(output_dir),
        samples=len(config.samples),
        files=sum(len(paths) for paths in fileset.values()),
        regions=len(config.regions),
        workers=workers,
        chunksize=chunksize,
        maxchunks=maxchunks,
        stage_dir=str(stage_dir) if stage_dir is not None else None,
        staged_bytes=staged_bytes,
        stage_seconds=stage_seconds,
        wall_seconds=elapsed,
    )
    metadata_path = output_dir / "Histograms" / "coffea-run.json"
    metadata_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    return summary
