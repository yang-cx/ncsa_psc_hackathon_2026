#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# StatAnalysis 0.8.2; digest resolved from the tested registry image.
IMAGE = "gitlab-registry.cern.ch/atlas/statanalysis@sha256:36a8c06ae90401e3629830c8ffe3b9fcf0b2a1841b28f59e4ecfa49c243051e1"
CONTAINER_ENGINE = "podman-hpc"

PROJECT_DIR = Path(os.environ.get("ATLASRL_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()
CONTAINER_PROJECT_DIR = "/workdir"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def to_container_path(path: str | Path) -> str:
    """
    Convert a project-local path to the corresponding path inside the container.

    Example:
      configs/myfit.config -> /workdir/configs/myfit.config
    """
    path = Path(path)

    if path.is_absolute():
        path = path.resolve()
        try:
            rel = path.relative_to(PROJECT_DIR)
        except ValueError as exc:
            raise ValueError(
                f"Path {path} is outside PROJECT_DIR={PROJECT_DIR}. "
                "Move it under the project directory or add another mount."
            ) from exc
    else:
        rel = path

    return str(Path(CONTAINER_PROJECT_DIR) / rel)

def discover_symlink_target_mounts() -> list[tuple[str, str]]:
    """
    Find symlinks inside PROJECT_DIR whose targets live outside PROJECT_DIR.

    Mount each external target at the same absolute path inside the container,
    so absolute symlinks continue to work:

        inputs/Data -> /global/.../gamgam_data

    becomes usable because /global/.../gamgam_data is mounted into the container.
    """
    mounts: set[tuple[str, str]] = set()
    project = PROJECT_DIR.resolve()

    for link in PROJECT_DIR.rglob("*"):
        if not link.is_symlink():
            continue

        raw_target = os.readlink(link)
        literal_target = Path(raw_target)
        if not literal_target.is_absolute():
            literal_target = (link.parent / literal_target).resolve()
        resolved_target = link.resolve()

        if not resolved_target.exists():
            continue

        try:
            resolved_target.relative_to(project)
            continue  # target is already inside project mount
        except ValueError:
            pass

        for target in {literal_target, resolved_target}:
            mount_path = target if resolved_target.is_dir() else target.parent
            mounts.add((str(mount_path), str(mount_path)))

    return sorted(mounts)


def container_cmd(
    shell_command: str, *, discover_external_mounts: bool = True
) -> list[str]:
    """
    Build a non-interactive podman-hpc command for TRExFitter.

    - no -it from Python
    - project is mounted as /workdir
    - external symlink targets are mounted at their original absolute paths
    - --group-add keep-groups preserves NERSC project group permissions
    """
    mounts = [(str(PROJECT_DIR), CONTAINER_PROJECT_DIR, "rw")]

    if discover_external_mounts:
        for src, dst in discover_symlink_target_mounts():
            mounts.append((src, dst, "ro"))

    cmd = [
        CONTAINER_ENGINE,
        "run",
        "--rm",
        "--group-add",
        "keep-groups",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/.cache",
    ]

    seen = set()
    for src, dst, mode in mounts:
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        cmd.extend(["-v", f"{src}:{dst}:{mode}"])

    cmd.extend(
        [
            "-w",
            CONTAINER_PROJECT_DIR,
            IMAGE,
            "bash",
            "-lc",
            f"umask 0002; {shell_command}",
        ]
    )

    return cmd

def run(cmd: list[str], label: str, log_dir: Path | None = None) -> None:
    """Run a command while streaming output and retaining separate log files."""
    print(f"\n=== {label} ===", flush=True)
    print(quote_cmd(cmd), flush=True)

    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout: list[str] = []
    stderr: list[str] = []

    def copy_stream(source, destination, collected: list[str]) -> None:
        for line in iter(source.readline, ""):
            collected.append(line)
            print(line, end="", file=destination, flush=True)
        source.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(copy_stream, process.stdout, sys.stdout, stdout),
            executor.submit(copy_stream, process.stderr, sys.stderr, stderr),
        ]
        returncode = process.wait()
        for future in futures:
            future.result()

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "_")
        (log_dir / f"{safe_label}.stdout.log").write_text("".join(stdout))
        (log_dir / f"{safe_label}.stderr.log").write_text("".join(stderr))

    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")


def run_capture(cmd: list[str], label: str, log_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {label} ===")
    print(quote_cmd(cmd))

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "_").replace(",", "_")
        (log_dir / f"{safe_label}.stdout.log").write_text(result.stdout)
        (log_dir / f"{safe_label}.stderr.log").write_text(result.stderr)

    return result


def read_config_job_name(config: str | Path) -> str | None:
    pattern = re.compile(r'^\s*Job:\s*"([^"]+)"')
    for line in Path(config).read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def read_config_regions(config: str | Path) -> list[str]:
    pattern = re.compile(r'^\s*Region:\s*"([^"]+)"')
    regions: list[str] = []
    for line in Path(config).read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            regions.append(match.group(1))
    return regions


def read_config_readfrom(config: str | Path) -> str | None:
    pattern = re.compile(r"^\s*ReadFrom:\s*(\S+)")
    for line in Path(config).read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip('"').upper()
    return None


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_significance_from_text(text: str) -> float | None:
    patterns = [
        r"SIGNIFICANCE\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Observed\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"obs(?:erved)?\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"Z0\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_significance_from_outputs(config: str | Path, log_dir: str | Path) -> float | None:
    candidates: list[Path] = []
    config_path = Path(config)
    job_name = read_config_job_name(config_path)

    for path in [Path(log_dir), config_path.parent]:
        if path.exists():
            candidates.extend(path.rglob("*.log"))
            candidates.extend(path.rglob("*.txt"))
            candidates.extend(path.rglob("*.json"))

    if job_name:
        output_dir = PROJECT_DIR / job_name
        if output_dir.exists():
            candidates.extend(output_dir.rglob("*.log"))
            candidates.extend(output_dir.rglob("*.txt"))
            candidates.extend(output_dir.rglob("*.json"))

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(errors="ignore"))
                if "significance" in data:
                    return float(data["significance"])
            significance = parse_significance_from_text(path.read_text(errors="ignore"))
        except Exception:
            continue
        if significance is not None:
            return significance

    return None


def check_setup() -> None:
    shell_command = (
        "echo 'Container:' && hostname && "
        "echo 'Working directory:' && pwd && "
        "echo 'User/groups:' && id && "
        "echo 'trex-fitter path:' && which trex-fitter && "
        "echo 'Files in /workdir:' && ls -la /workdir"
    )

    run(container_cmd(shell_command), label="check setup")


def run_trex_action(
    action: str,
    config: str | Path,
    log_dir: str | Path = "trex_logs",
    regions: list[str] | None = None,
) -> None:
    config_in_container = to_container_path(config)

    shell_parts = ["trex-fitter", shlex.quote(action), shlex.quote(config_in_container)]
    if regions:
        shell_parts.append(shlex.quote(f'Regions="{",".join(regions)}"'))
    shell_command = " ".join(shell_parts)

    run(
        container_cmd(shell_command),
        label=f"trex-fitter {action}" + (f" Regions={','.join(regions)}" if regions else ""),
        log_dir=Path(log_dir),
    )


def run_trex_action_capture(
    action: str,
    config: str | Path,
    log_dir: str | Path = "trex_logs",
    regions: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    config_in_container = to_container_path(config)
    shell_parts = ["trex-fitter", shlex.quote(action), shlex.quote(config_in_container)]
    if regions:
        shell_parts.append(shlex.quote(f'Regions="{",".join(regions)}"'))
    shell_command = " ".join(shell_parts)
    label = f"trex-fitter {action}" + (f" Regions={','.join(regions)}" if regions else "")
    return run_capture(container_cmd(shell_command), label=label, log_dir=Path(log_dir))


def run_trex_n_parallel_regions(
    config: str | Path,
    log_dir: str | Path = "trex_logs",
    workers: int | None = None,
    regions_per_job: int = 1,
    regions: list[str] | None = None,
) -> None:
    all_regions = regions or read_config_regions(config)
    if not all_regions:
        raise RuntimeError(f"No Region blocks found in {config}")

    jobs = chunked(all_regions, max(1, regions_per_job))
    if workers is None or workers <= 0:
        workers = len(jobs)
    workers = max(1, min(workers, len(jobs)))

    print(
        f"Running trex-fitter n in parallel: {len(all_regions)} regions, "
        f"{len(jobs)} jobs, {workers} workers"
    )

    failures: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_regions = {
            executor.submit(
                run_trex_action_capture,
                "n",
                config,
                log_dir,
                region_group,
            ): region_group
            for region_group in jobs
        }

        for future in as_completed(future_to_regions):
            region_group = future_to_regions[future]
            result = future.result()
            if result.returncode == 0:
                print(f"Finished n for Regions={','.join(region_group)}")
            else:
                print(f"FAILED n for Regions={','.join(region_group)} exit={result.returncode}")
                if result.stderr:
                    print(result.stderr[-4000:])
                failures.append((region_group, result))

    if failures:
        failed = [".".join(group) for group, _ in failures]
        raise RuntimeError(f"Parallel n failed for region groups: {failed}")


def run_trex_sequence(
    actions: list[str],
    config: str | Path,
    log_dir: str | Path = "trex_logs",
    parallel_regions: bool = True,
    region_workers: int | None = None,
    regions_per_job: int = 1,
) -> None:
    read_from = read_config_readfrom(config)
    for action in actions:
        if action == "h" and read_from != "HISTO":
            print(f'\n=== skipping trex-fitter h ===\nReadFrom is {read_from or "unset"}, not HISTO.')
            continue
        if action == "n" and parallel_regions:
            run_trex_n_parallel_regions(
                config,
                log_dir=log_dir,
                workers=region_workers,
                regions_per_job=regions_per_job,
            )
        else:
            run_trex_action(action, config, log_dir=log_dir)


def main() -> None:
    global PROJECT_DIR

    parser = argparse.ArgumentParser(
        description="Run preinstalled TRExFitter from StatAnalysis image using podman-hpc."
    )

    parser.add_argument(
        "config",
        help="TRExFitter config file relative to the project directory.",
    )

    parser.add_argument(
        "--actions",
        nargs="+",
        default=["n", "w", "f", "s"],
        help="TRExFitter actions to run, e.g. --actions n w f s l d. Use h only for HISTO configs.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check container setup and exit.",
    )

    parser.add_argument(
        "--log-dir",
        default="trex_logs",
        help="Directory where stdout/stderr logs are saved.",
    )

    parser.add_argument(
        "--parallel-regions",
        action="store_true",
        dest="parallel_regions",
        default=True,
        help='Split action "n" into parallel jobs using TRExFitter Regions="..."; enabled by default.',
    )

    parser.add_argument(
        "--no-parallel-regions",
        action="store_false",
        dest="parallel_regions",
        help='Run action "n" as one serial TRExFitter job instead of splitting by region.',
    )

    parser.add_argument(
        "--region-workers",
        type=int,
        default=0,
        help="Maximum number of parallel region jobs for action n. Use 0 to match the number of region jobs.",
    )

    parser.add_argument(
        "--regions-per-job",
        type=int,
        default=1,
        help="Number of regions grouped into each parallel n job.",
    )

    parser.add_argument(
        "--project-dir",
        default=str(PROJECT_DIR),
        help="Project directory mounted into the TRExFitter container.",
    )

    args = parser.parse_args()

    PROJECT_DIR = Path(args.project_dir).resolve()

    if args.check:
        check_setup()
        return

    run_trex_sequence(
        args.actions,
        args.config,
        log_dir=args.log_dir,
        parallel_regions=args.parallel_regions,
        region_workers=args.region_workers,
        regions_per_job=args.regions_per_job,
    )
    significance = parse_significance_from_outputs(args.config, args.log_dir)
    if significance is not None:
        print(f"SIGNIFICANCE={significance}")
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.log_dir) / "results.json").write_text(
            json.dumps({"significance": significance}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
