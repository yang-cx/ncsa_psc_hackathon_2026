# Python layout and entry points

Run commands from the repository root through the locked uv environment.

| Purpose | Preferred command/module | Existing entry point retained |
| --- | --- | --- |
| Unified static and optional input verification | `python -m trex_fitter.config_verify` | Same command |
| Native/Coffea workflow | `python -m trex_fitter.runner` | `python trex_fitter/runner.py` |
| JSON evaluation result | `python -m trex_fitter.evaluate` | `python trex_fitter/scripts/evaluate_config.py` |
| Deterministic mock | `python -m trex_fitter.mock` | `python trex_fitter/scripts/mock_trex.py` |
| Legacy native runner, including parallel regions | `python -m trex_fitter.runtime` | `python trex_fitter/scripts/trex.py` |

## Code ownership

- `config_format.py`: shared config syntax and errors; no execution backend
  dependencies. Old parser imports from `coffea_backend.config` are re-exported.
- `config_verify.py`, `schema.py`, `semantics.py`, `input_checks.py`: unified
  report and validation stages. Native schema snapshots live in `schemas/`.
- `runtime.py`: container commands, subprocess logging, result extraction and
  the legacy native CLI. `runner.py` imports it directly rather than importing
  a script through a modified search path.
- `runner.py`: orchestration and backend selection.
- `evaluate.py`, `mock.py`: evaluation/reporting and its deterministic fixture.
- `coffea_backend/`: histogram-specific config interpretation, expression
  evaluation, processing, writing, comparison and capability checking.
- `scripts/`: compatibility wrappers and the still-used Hyy fetch script.
- `sft/`: reviewed example data; `tests/trex_fitter/`: regression tests.

## Legacy interfaces reviewed

`verifier.py` is the older structural-only preflight. Its output contract and
accepted language differ from the unified verifier, so it remains available
for compatibility; new callers should use `config_verify`.

`coffea_backend.verify` is the capability checker used internally by the
unified verifier and referenced by an older SFT seed. Its CLI is retained.

The native parallel-region runner has behavior absent from the primary runner,
so it is not removed. The mock is used by evaluation tests, and
`scripts/fetch_hyy_inputs.sh` is referenced by the data documentation.

No existing commands were deleted or redirected to different validation
semantics in this restructuring. Compatibility wrappers share the same module
objects, including legacy module-level configuration. Generated reports,
ROOT files, logs and smoke-test outputs stay under ignored `artifacts/`.

Training and inference retain their separate environment and model-runtime
modules; this restructuring does not change their interfaces or dependencies.
