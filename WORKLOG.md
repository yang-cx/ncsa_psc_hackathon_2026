# Worklog

This file records completed work, decisions, measurements, and open questions
for the TRExFitter Hyy histogramming effort.

## 2026-09-08 — Repository setup, baseline, and implementation plan

### Objective

Establish a reproducible TRExFitter Hyy baseline, identify the histogramming
bottleneck, and plan a faster histogram producer using Coffea and potentially
atlas-schema. Feature development is paused until the plan is approved.

### Repository and remotes

- Updated the local repository to upstream `origin/main` at
  `3f60379d7ef1640df4b25050bd19815c581175ab`.
- Initialized recursive submodules:
  - `data/samples` at `6ba62e28d7536fa2fb1232d32777b68c4283636b`.
  - `verl` at `c2429f29a25d573f63d9bcc29e7ceb690817dce9`.
- Added the personal fork as remote `fork`:
  `https://github.com/yang-cx/ncsa_psc_hackathon_2026.git`.
- Configured `remote.pushDefault=fork`; upstream remains `origin`.
- The repository remains on `main`. No feature branch, feature implementation,
  commit, or push has been made for the Coffea work.

### Environment findings

- `atlas_local` loads ATLAS Local Root Base on the host, but the resulting host
  environment does not provide the required TRExFitter executable.
- `atlas_shifter` enters a generic ATLAS AlmaLinux 9 Shifter environment, which
  also does not contain the repository's pinned TRExFitter setup.
- The working path for this repository is:
  1. load NERSC Python 3.11 on the host;
  2. run `trex_fitter/runner.py`;
  3. let the runner use Podman-HPC with the pinned StatAnalysis image.
- Tested analysis environment:
  - StatAnalysis 0.8.2;
  - TRExFitter v1.10.0;
  - ROOT v6.40.04;
  - container Python 3.10.6;
  - image digest
    `sha256:36a8c06ae90401e3629830c8ffe3b9fcf0b2a1841b28f59e4ecfa49c243051e1`.

### Input provenance and inventory

- The canonical inputs are now provided by the `data/samples` Git submodule,
  whose remote is the Hugging Face dataset repository
  `datasets/ho22joshua/hackathon_samples`.
- Hyy inputs are under `data/samples/hyy`:
  - 26 ROOT files;
  - 11,584,796,257 bytes (10.79 GiB);
  - 41,541,644 events;
  - 36,564,144 data events and 4,977,500 MC events;
  - tree name `analysis`.
- A manual download made before the submodule update was preserved at
  `data/samples.local-inputs-20260908/`. It duplicates about 11 GiB of data and
  has not been deleted.
- Incomplete results from two interrupted attempts were archived under the
  ignored artifact tree as:
  - `artifacts/trex_fitter/interrupted/hyy-pre-pull-20260908/`;
  - `artifacts/trex_fitter/interrupted/hyy-post-pull-20260908/`.

### Completed TRExFitter baseline

- Slurm job: `58076459` on `nid004083`.
- Result: completed successfully; `n`, `w`, `f`, and `s` all exited 0.
- Slurm allocation wall time: 34:59.
- Per-action wall time:

| Action | Purpose | Wall time |
|---|---|---:|
| `n` | Ntuple reading and histogram production | 32:06.06 |
| `w` | Workspace construction | 0:59.12 |
| `f` | Fit | 0:52.41 |
| `s` | Significance | 0:53.62 |

- Histogram production accounts for 92.1% of the measured action time.
- Peak Slurm RSS: 13,542,092 KiB (12.91 GiB).
- Total Slurm CPU time: 30:01.042.
- Complete archived Hyy output size: 578,526 bytes.
- Histogram output size: 145,353 bytes.
- Completed output location:
  `artifacts/trex_fitter/baseline-20260908/output/hyy/`.

Detailed report: [TRExFitter Hyy baseline performance report](artifacts/trex_fitter/baseline-20260908/performance-report.md).

### Fit and significance reference

- Minuit status: 0.
- Hessian status: 0.
- NLL: 26,463.19978.
- `mu_H = 0.773007 ± 0.438333`.
- Expected significance for `mu=1`: 2.39328 sigma.
- Observed significance: 1.86253 sigma.
- Expected p-value: 0.00834924.
- Observed p-value: 0.0312645.
- Reported bad fits: 0.

Fit output:
[hyy/Fits/hyy.txt](artifacts/trex_fitter/baseline-20260908/output/hyy/Fits/hyy.txt).

### Histogram reference and warnings

- TRExFitter produced one ROOT file per region, archived under
  `artifacts/trex_fitter/baseline-20260908/output/hyy/Histograms/`.
- Each sample has raw `_orig`, post-processed nominal, and `_regBin` `TH1D`
  objects below `<region>/<sample>/nominal/`.
- The configured histogram has 15 bins from 100 to 160 GeV, so the actual bin
  width is 4 GeV. The config comment claiming 2 GeV bins is inconsistent with
  the configured bin count.
- The raw `cat_2jet_diphoton` second bin is zero. TRExFitter warns and adjusts
  the post-processed nominal bin to a small nonzero value.
- Significance extraction prints a NaN diagnostic, but all fit statuses are 0,
  both reported significances are finite, and the tool reports zero bad fits.
- All four captured stderr logs are empty.

### Performance diagnosis

TRExFitter rereads all 26 ROOT files independently for each of the six Hyy
regions. Each region scan takes about 4:37--5:29. This corresponds to roughly
249 million region-event evaluations and 64.74 GiB of logical compressed-file
scans for a 10.79 GiB dataset.

The primary proposed optimization is to read each file/chunk once, calculate
the common variables once, evaluate all six category masks, and fill all region
histograms in the same pass. Removing the repeated scans presents an inferred
nearly sixfold I/O opportunity before adding parallel execution; this is not
yet a measured Coffea speedup.

### Proposed Coffea work — planning only

Planning document: [Coffea histogramming plan](artifacts/trex_fitter/coffea-histogramming-plan.md).

Recommended first milestone:

1. Validate one data and one MC file with Coffea and atlas-schema in a
   disposable, pinned environment.
2. Treat atlas-schema as optional because these are legacy flat ntuples and
   photon/jet transverse momenta are already in GeV.
3. Define a typed Hyy-specific analysis specification instead of starting with
   a general parser for arbitrary TRExFitter C++ expressions.
4. Implement a deterministic single-worker producer that preserves `sumw`,
   `sumw2`, bin edges, zero bins, and flow-bin policy.
5. Write ROOT histograms and a `ReadFrom: HIST` TRExFitter adapter.
6. Require bin-by-bin, yield, workspace, fit, and significance equivalence to
   the completed baseline.
7. Only after equivalence, benchmark Dask scaling on NERSC.

### Artifact policy

- Generated run outputs, logs, timing files, reports, config snapshots, and
  interrupted results are consolidated under `artifacts/`.
- `artifacts/` is already excluded by the repository's `.gitignore`.
- Source code and future committed documentation remain outside that tree.
- Input data and the preserved manual input backup are not classified as
  generated artifacts and were not moved by this cleanup.

### Approval gates before implementation

- Approve an Hyy-specific first milestone rather than a generic TRExFitter
  expression translator.
- Approve a fallback to Coffea's base schema/direct Uproot forms if
  atlas-schema does not cleanly support the legacy ntuples.
- Confirm that nominal-only histograms are sufficient for the first milestone;
  the current Hyy config has no histogram shape systematics.
- Keep the current 15-bin/4-GeV behavior for equivalence, or separately approve
  a physics change to the commented 2-GeV intent.
- Decide separately whether to retain or remove the duplicate manual input copy
  and interrupted output directories.

After approval, the proposed branch is `feature/coffea-histogramming`, created
from the then-current upstream `main`, with all pushes going to `fork`.

## 2026-09-09 — Coffea `trex-fitter n` backend implementation

### Repository state

- Pulled the current upstream `origin/main` at
  `9031476db58dbbc88bbb4f984660d8af9aa59e3b`.
- Initialized the newly added recursive submodule
  `data/datasets/root-sft-dataset`.
- Created and developed on `feature/coffea-histogramming`.
- Kept `fork` (`yang-cx/ncsa_psc_hackathon_2026`) as the push target.

### Implemented interface

The native TREx `.config` remains the single analysis configuration. The
runner now accepts:

```bash
python3 trex_fitter/runner.py data/configs/examples/hyy.config \
  --backend coffea --actions n
```

The default `--backend trex` behavior is unchanged. With the Coffea backend,
only `n` is intercepted; requested `w`, `f`, and `s` actions continue in
the existing pinned StatAnalysis/TRExFitter container.

Implemented components:

- native TREx `Job`, `Region`, and `Sample` block parser;
- safe AST-based vectorized expression evaluator (no Python `eval`);
- ROOT-style boolean operators, jagged indexing, arithmetic, comparisons, and
  the mathematical functions used by Hyy;
- Coffea `Runner` with iterative or futures executors;
- one chunk pass that materializes each needed branch once and fills all six
  regions;
- optional Coffea `BaseSchema` or `atlas-schema` `NtupleSchema`;
- weighted histograms preserving bin contents and `sumw2`;
- TREx-compatible split ROOT hierarchy, flow folding, background empty-bin
  repair, and `Binnings/*.txt`;
- a 126-object bin-content/variance comparison command;
- node-local input staging and artifact-contained downstream TREx working
  directories;
- pinned optional dependencies, user documentation, and ten unit tests.

The current first milestone intentionally rejects `Systematic` blocks with a
clear error. The Hyy config is nominal-only, so this does not reduce its
coverage. General TREx configuration-language compatibility remains future
work.

### Environment and compute

- Interactive CPU allocation: Slurm `58122048`, node `nid004157`.
- NERSC Python 3.11.7 from `python/3.11-24.1.0`.
- Disposable node-local environment: `/tmp/coffea-venv-58122048`.
- Direct dependency pins:
  - `coffea[dask]==2025.7.0`;
  - `atlas-schema==0.5.0`.
- Full run: eight `FuturesExecutor` workers, 250,000 events per chunk.

An unstaged diagnostic showed Coffea's branch-oriented lazy reads waiting on
the DVS project filesystem. The production command therefore staged all input
ROOT files once to node-local tmpfs. This took 18.61 seconds and is included in
the end-to-end timing. Once staged, all eight workers ran at sustained CPU.

### Full-run performance

| Measurement | Original TREx `n` | Coffea `n` |
|---|---:|---:|
| End-to-end wall | 32:06.06 | 1:36.56 |
| Speedup | — | 19.95× |
| Wall-time reduction | — | 94.99% |
| Node-local staging | none | 18.61 s, included |
| Coffea application wall | — | 1:33.84 |
| Processing/output after staging | — | 1:15.23 |
| Chunks | six complete region scans | 172, all regions per chunk |

The Coffea timed command used 571.23 user seconds, 15.62 system seconds, 607%
average CPU, and at most 441,092 KiB RSS for an individual process.

Detailed report:
[Coffea performance and equivalence report](artifacts/trex_fitter/coffea-full-staged-20260909/performance-report.md).

### Histogram and schema equivalence

- Compared all 126 `TH1D` objects against the archived TREx `n` output.
- Result: 126 passed, 0 failed at `rtol=1e-6`, `atol=1e-8`.
- Worst relative content difference: `3.2459139341843383e-15`.
- Worst relative variance difference: `3.574117739342478e-15`.
- Every path and bin edge matched.
- A one-chunk-per-sample `atlas-schema` smoke output was bit-for-bit identical
  to the corresponding `BaseSchema` output across all 126 objects.

### Downstream TREx validation

Unmodified TRExFitter `w`, `f`, and `s` consumed the Coffea output from
the ignored artifact tree. All actions exited 0, and all stderr logs are empty.

| Quantity | Coffea histograms + TREx |
|---|---:|
| Minuit / Hessian status | 0 / 0 |
| NLL | 26,463.19978 |
| `mu_H` | 0.773007 ± 0.438330 |
| Expected / observed significance | 2.39328 / 1.86253 |
| Expected / observed p-value | 0.00834924 / 0.0312645 |
| Reported bad fits | 0 |

These match the original baseline at the reported precision. The downstream
`w/f/s` wrapper took 6:17.04 on the interactive allocation, dominated by
three separate Podman-HPC container startups. Combining it with Coffea `n`
gives 7:53.60, 4.42× faster than the baseline measured-action sum.

### Artifact locations

- Full Coffea and downstream output:
  `artifacts/trex_fitter/coffea-full-staged-20260909/output/hyy/`.
- Numerical comparison:
  `artifacts/trex_fitter/coffea-full-staged-20260909/comparison.json`.
- Performance report:
  `artifacts/trex_fitter/coffea-full-staged-20260909/performance-report.md`.
- Full run timing/log:
  `artifacts/trex_fitter/coffea-full-staged-20260909/time.txt` and
  `run.log`.
- Downstream timing/logs:
  `artifacts/trex_fitter/coffea-full-staged-20260909/trex-wfs-time.txt`,
  `trex-wfs.log`, and `logs/`.

All generated outputs remain under ignored `artifacts/`. The node-local stage
and virtual environment are disposable and will be removed automatically when
the interactive allocation ends.

## 2026-09-10 — Duplicate input cleanup

- Removed the untracked `data/samples.local-inputs-20260908/` safety backup at
  the user's request, reclaiming approximately 11 GiB.
- Kept the canonical `data/samples` Git submodule and its Hyy inputs unchanged.

## 2026-09-10 — Container version verification

- Queried the repository's exact pinned StatAnalysis image by immutable digest.
- Confirmed StatAnalysis 0.8.2, TRExFitter v1.10.0, ROOT v6.40.04, container
  Python 3.10.6, and xRooFit `v0.0.4-22-g5eb77d8`.
- Checked out the matching `v1.10.0` release tag in the external
  `TRExFitter-Documentation` clone at commit
  `ee86eaa730325cb30534b1032dfe42c917a37930`.
- Documented the image digest, installed executable path, and host/container
  responsibility split in `trex_fitter/README.md`.

## 2026-09-10 — Source-audited minimal backend and SFT seed

- Audited the nominal `n` path against TRExFitter v1.10.0 source, especially
  `NtupleReader::GetHistogram`, `FullSelection`, `FullWeight`, flow folding,
  `SampleHist::FixEmptyBins`, and ROOT histogram writing.
- Kept the backend deliberately Hyy-specific instead of expanding toward the
  full TRExFitter configuration language.
- Added a Pydantic-backed static compatibility verifier that does not open ROOT
  inputs or invoke TRExFitter. It validates typed Job/Region/Sample basics,
  fixed binning, safe expression syntax, required MC weights, unique names,
  and the explicit feature boundary.
- The Hyy config passes with zero errors. Its 48 warnings identify presentation
  settings and Fit/NormFactor blocks that nominal histogramming does not use.
- Corrected two source-audit discrepancies: exact v1.10 empty-bin error
  handling and ROOT histogram/x-axis titles. Strengthened differential checks
  to include those titles.
- Expanded the unit suite from 10 to 18 tests; all pass under NERSC Python
  3.11 with the pinned Coffea environment.
- Completed a full 26-file/41.5-million-event validation on Slurm job
  `58151768`. All 126 ROOT histograms match the native TRExFitter baseline,
  with maximum relative content and variance differences of approximately
  `3.25e-15` and `3.57e-15`.
- Measured 1:36.19 end-to-end with a cold 10.79 GiB node-local stage and
  1:17.06 with the stage reused, versus 32:06.06 for the native baseline.
- Added a source-referenced atomic-operation map and eight hand-reviewed SFT
  seed examples covering typical analyst actions and explicit unsupported
  cases. These are examples for review, not yet a production training corpus.
- Stored all generated output and the detailed performance report under the
  ignored `artifacts/trex_fitter/coffea-source-audit-20260910/` directory.

## 2026-09-10 — uv environment management

- Adopted `uv` as the single source of truth for host-side Python dependencies.
- Added a root `pyproject.toml`, Python 3.11 selector, and committed `uv.lock`;
  removed the older hand-maintained Coffea requirements file.
- Kept the default environment limited to the static verifier's Pydantic,
  NumPy, and Awkward dependencies. Coffea and atlas-schema are separate optional
  extras, while TRExFitter and ROOT remain exclusively in the container.
- Generated the lock with NERSC Python 3.11.7 and uv 0.8.0 on Slurm job
  `58164000`.
- Verified `uv sync --locked` and the Hyy static verifier with the default
  environment. It occupies approximately 82 MiB on the tested node.
- Verified the full `coffea` plus `atlas-schema` extras and all 19 tests. The
  complete environment occupies approximately 815 MiB; its size is primarily
  Coffea's scientific and distributed-I/O dependency graph, not ROOT or ML
  frameworks.
