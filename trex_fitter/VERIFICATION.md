# Config verification

One command always checks native schema validity, supported static semantics,
and Coffea compatibility. It opens no ROOT files by default:

```bash
uv run --locked python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config --actions nwsf
```

`--actions n w s f` and `--actions nwsf` are equivalent. The default is `nwfs`.
Use `--actions n` when only histogramming is planned. Native TRExFitter executes
combined actions in its own internal order, regardless of letter order.

## Default criteria

- Every supplied setting is checked against the pinned v1.10.0 native schema.
- Basic Job/Fit/Region/Sample/NormFactor values and ranges are checked.
- Fit actions (`f`, `l`, `s`, `r`, `i`, `x`) require a non-validation region.
- `POIAsimov` must be a finite scalar for one POI, or declared `name@value` pairs.
- POIs can resolve to norm-factor blocks, inline sample norm factors, nuisance
  parameters (including `alpha_` names), template parameters, shape-expression
  parameters, and EFT parameters from `EFTValue` for nonsplit EFT fits.
- Sample, norm-factor, shape-factor, and systematic references are checked;
  `all` and `none` are recognized. Common systematic drop/keep references are
  also checked.
- Norm-factor attachments must be unique per sample/region. Repeated names in
  disjoint attachments are allowed.
- `NumCPU`, `ToysHistoNbins`, `RankingNPfraction`, and configured scan step
  counts are checked. Ranking lists must match the number of POIs.
- Coffea selection/weight syntax and supported histogram operations are checked
  independently. A valid native setting need not be supported by Coffea.

Rules have stable diagnostic codes. The new semantic diagnostics include
source references into `Root/ConfigReader.cc` at pinned commit
`52e62c30a1faf1ca9fdfe0db74160bb9e00e86e9`.

Scan step ranges are deliberately stricter than native behavior: TRExFitter
warns and replaces out-of-range values with defaults; this verifier reports
an error so generated configs must state a usable value explicitly.

## Optional input checks

```bash
uv run --locked --extra inputs python -m trex_fitter.config_verify \
  data/configs/examples/hyy.config --check-inputs --actions nwsf \
  --json artifacts/trex_fitter/verification.json
```

The `inputs` extra supplies Uproot; the existing `coffea` extra also supplies it.
These checks inspect metadata and histogram axes, without reading event arrays:

- For the supported Coffea NTUP subset, resolve every configured file pattern,
  check each ROOT file and tree, and check branches used by selections,
  weights, and region variables.
- For simple nominal HIST configs, check files, named TH1 objects, and matching
  edges across samples within a region. One unambiguous HistoPath/HistoFile/
  HistoName per sample-region is required.
- Unsupported input layouts receive an `input_coverage` error. Friend trees,
  advanced suffix combinations, systematic variations, and generated inputs
  are not claimed as checked.

`--project-dir` specifies the host root corresponding to `/workdir`. Its
`data/samples` maps to `/workdir/inputs`; the repository root is the default.
Input checks are read-only but their time depends on the number of files and
filesystem latency. No file sampling is performed.

The JSON/Python report contains `analysis_valid`, `coffea_compatible`, and
`inputs_valid` (`null` when not requested), plus separate diagnostics and
`input_files_checked`. Overall `valid` requires both default checks and, when
requested, passing input checks. Thus a native HIST analysis can have
`analysis_valid=true`, `inputs_valid=true`, but `coffea_compatible=false`.

## Differences from ReadFullConfig

Native `ReadFullConfig` builds an effective analysis, applying defaults,
inheritance, name normalization, command-line overrides, generated parameters,
and template expansions. This verifier does not build that complete model.
It does not yet reproduce full MultiFit, unfolding, EFT splitting, morphing,
shape-factor bin consistency, all sample arithmetic/dependency checks, or every
conditional rule in `PostConfig`. Some inherited input configurations remain
outside the basic static profile.

Metadata checks do not validate ROOT-only formulas, runtime array shapes,
selection yields, numerical fit stability, or physics intent. Native parsing
and actual execution remain necessary for those cases. No native-parser or
event-sampling mode is included here.
