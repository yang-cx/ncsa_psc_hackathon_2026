# TRExFitter runner

`runner.py` runs the example analysis configs with the pinned StatAnalysis
container. For configs that read ntuples, it can alternatively use Coffea for
the expensive histogramming (`n`) action and pass the resulting ROOT
histograms to unmodified TRExFitter `w`, `f`, and `s` actions.

The native TREx `.config` is the source of truth for both backends. The Coffea
backend currently supports the subset exercised by `hyy.config`: `Job`,
`Region`, and `Sample` definitions, flat or jagged branch expressions, nominal
MC weights, and split histogram files. It fails explicitly on `Systematic`
blocks rather than silently producing incomplete histograms.

## Environment

On a NERSC compute node, create an environment on node-local storage for a
one-off run (or use a persistent venv under `$PSCRATCH`):

```bash
module load python/3.11-24.1.0
python -m venv /tmp/trex-coffea
/tmp/trex-coffea/bin/python -m pip install -r trex_fitter/coffea-requirements.txt
```

The direct dependencies are pinned in `coffea-requirements.txt`. The default
`base` schema exposes the original branch names exactly as TREx expressions
spell them. `--coffea-schema atlas` enables `atlas-schema` collection grouping;
the expression resolver understands both layouts.

## Run

Run only the Coffea replacement for `trex-fitter n`:

```bash
/tmp/trex-coffea/bin/python trex_fitter/runner.py \
  data/configs/examples/hyy.config \
  --backend coffea --actions n --coffea-workers 8 \
  --coffea-stage-dir /tmp/trex-hyy-$SLURM_JOB_ID
```

This writes `hyy/Histograms/hyy_<region>_histos.root`, including the `_orig`,
nominal, and `_regBin` objects that the later TREx actions expect. Running all
stages uses Coffea only for `n`:

```bash
/tmp/trex-coffea/bin/python trex_fitter/runner.py \
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
/tmp/trex-coffea/bin/python -m trex_fitter.coffea_backend.compare \
  artifacts/trex_fitter/baseline-20260908/output/hyy \
  artifacts/trex_fitter/coffea-full/output/hyy \
  --json artifacts/trex_fitter/coffea-full/comparison.json
```

Compatibility means identical ROOT paths and bin edges plus numerically
consistent bin contents and variances. ROOT object titles and other cosmetic
metadata are not part of the contract.
