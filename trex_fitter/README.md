# TRExFitter runner

`runner.py` runs the example analysis configs with the pinned StatAnalysis
container. For configs that read ntuples, it can alternatively use Coffea for
the expensive histogramming (`n`) action and pass the resulting ROOT
histograms to unmodified TRExFitter `w`, `f`, and `s` actions.

The native TREx `.config` is the source of truth for both backends. The Coffea
backend intentionally supports the basic subset exercised by `hyy.config`: `Job`,
`Region`, and `Sample` definitions, flat or jagged branch expressions, nominal
MC weights, and split histogram files. It fails explicitly on `Systematic`
blocks rather than silently producing incomplete histograms.

## Environment

### Pinned TRExFitter container

The runner uses Podman-HPC with the immutable StatAnalysis 0.8.2 image:

```text
gitlab-registry.cern.ch/atlas/statanalysis@sha256:36a8c06ae90401e3629830c8ffe3b9fcf0b2a1841b28f59e4ecfa49c243051e1
```

The exact digest was inspected on NERSC on 2026-09-10 and contains:

- TRExFitter v1.10.0;
- ROOT v6.40.04;
- Python 3.10.6;
- xRooFit `v0.0.4-22-g5eb77d8`.

The container's `trex-fitter` executable is installed at:

```text
/usr/StatAnalysis/0.8.2/InstallArea/x86_64-el9-gcc14-opt/bin/trex-fitter
```

The image reference is defined in `trex_fitter/scripts/trex.py`. The traditional
backend runs `n`, `w`, `f`, and `s` in this container. The Coffea backend runs
only the replacement `n` implementation in the host-side Python environment;
the unmodified container still runs requested `w`, `f`, and `s` actions.
The corresponding TRExFitter documentation release is the `v1.10.0` tag of
`TRExStats/TRExFitter-Documentation` (commit
`ee86eaa730325cb30534b1032dfe42c917a37930`).

### Host-side Python environment

Host-side dependencies are managed by `uv` using the repository's
`pyproject.toml` and committed `uv.lock`. Load the NERSC Python module first;
`.python-version` selects the tested Python 3.11 series.

The default environment is intentionally small and supports static config
verification:

```bash
module load python
uv sync --locked
uv run --locked python -m trex_fitter.coffea_backend.verify \
  data/configs/examples/hyy.config
```

Install the optional Coffea histogramming stack on a compute node with:

```bash
uv sync --locked --extra coffea
```

Add `--extra atlas-schema` to both `uv sync` and `uv run` to enable the
optional atlas-schema collection layout. The default `base` schema exposes the
original branch names exactly as TREx expressions spell them. The expression
resolver understands both layouts. TRExFitter and ROOT remain in the pinned
StatAnalysis container and are not installed by `uv`.

On the tested NERSC Python 3.11 environment, the default verifier environment
occupies about 82 MiB. The complete Coffea plus atlas-schema environment is
about 815 MiB because Coffea brings the broader Scikit-HEP and distributed-I/O
stack. Neither environment includes ROOT, CUDA, or machine-learning frameworks.

## Run

Run only the Coffea replacement for `trex-fitter n`:

```bash
uv run --locked --extra coffea python trex_fitter/runner.py \
  data/configs/examples/hyy.config \
  --backend coffea --actions n --coffea-workers 8 \
  --coffea-stage-dir /tmp/trex-hyy-$SLURM_JOB_ID
```

This writes `hyy/Histograms/hyy_<region>_histos.root`, including the `_orig`,
nominal, and `_regBin` objects that the later TREx actions expect. Running all
stages uses Coffea only for `n`:

```bash
uv run --locked --extra coffea python trex_fitter/runner.py \
  data/configs/examples/hyy.config --backend coffea \
  --coffea-workers 8 \
  --coffea-stage-dir /tmp/trex-hyy-$SLURM_JOB_ID \
  --output-dir artifacts/trex_fitter/coffea-run/output
```

Use `--coffea-maxchunks 1` for a quick smoke test. Use `--output-dir` to keep a
benchmark or complete run under ignored `artifacts/`. When later actions are
requested, the runner starts TRExFitter in that output directory so it reads
the Coffea-produced Job directory in place.

`--coffea-stage-dir` is strongly recommended on NERSC. It performs one
sequential copy of each configured ROOT file to node-local storage before
Coffea starts its parallel, branch-oriented reads. Staging time is included in
the reported wall time, and complete files already present in the stage
directory are reused.

Compare a completed candidate against a TRExFitter reference:

```bash
uv run --locked --extra coffea python -m trex_fitter.coffea_backend.compare \
  artifacts/trex_fitter/baseline-20260908/output/hyy \
  artifacts/trex_fitter/coffea-full/output/hyy \
  --json artifacts/trex_fitter/coffea-full/comparison.json
```

Compatibility means identical ROOT paths, bin edges, ROOT histogram/axis
titles, and numerically consistent bin contents and variances.

## Fast static config verification

Check whether a config fits the supported nominal-NTUP subset without opening
ROOT files or starting TRExFitter:

```bash
uv run --locked python -m trex_fitter.coffea_backend.verify \
  data/configs/examples/hyy.config
```

The Pydantic-backed report separates errors from warnings. Histogram-affecting
unsupported features are errors; fit and normalization blocks used later by
unmodified TRExFitter are warnings. This verifies backend compatibility, not
the full TRExFitter language or the physics intent of an analysis.

The verified operation map is in
[`coffea_backend/ATOMIC_OPERATIONS.md`](coffea_backend/ATOMIC_OPERATIONS.md).
A small source-grounded SFT seed set is in [`sft/`](sft/).
