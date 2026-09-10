# Verified atomic operations

The Coffea backend is a fast executable specification of the nominal NTUP
operations needed by `data/configs/examples/hyy.config`. It is intentionally
not a second implementation of the full TRExFitter configuration language.
The static verifier rejects constructs that this backend cannot reproduce.

The source reference is TRExFitter v1.10.0, commit
`52e62c30a1faf1ca9fdfe0db74160bb9e00e86e9`.

| Atomic operation | Backend implementation | TRExFitter reference | Verification |
|---|---|---|---|
| Parse Job, Region, and Sample blocks | `config.py` | `Root/ConfigReader.cxx` | parser unit tests and Hyy config load |
| Verify the supported nominal subset | `verify.py` | configuration semantics below | Pydantic models and unsupported-field tests |
| Resolve nominal ntuple paths/files/tree | `config.py` | `Root/TRExFit.cc`, `FullNtuplePaths`, around line 7208 | Hyy input inventory and full run |
| Evaluate ROOT-style scalar/jagged expressions | `expressions.py` | `Root/Common.cc`, `TTree::Draw`, around lines 253–269 | isolated expression tests and histogram comparison |
| Apply sample and region selections | `processor.py` | `Root/TRExFit.cc`, `FullSelection`, around line 7070 | event processing and full Hyy comparison |
| Apply MC weights and luminosity; use unit data weight | `processor.py` | `Root/TRExFit.cc`, `FullWeight`, around line 7121; `Root/NtupleReader.cc` | full Hyy comparison |
| Fill fixed-bin weighted histograms with sumw2 | `processor.py` | `Root/NtupleReader.cc`, `GetHistogram`; `Root/Common.cc` | content and variance comparison |
| Fold underflow and overflow | `writer.py` | `Root/Common.cc`, around line 480 | isolated flow test |
| Repair non-positive background bins | `writer.py` | `Root/SampleHist.cc`, `FixEmptyBins`, around line 640 | edge-case unit tests and full Hyy comparison |
| Write nominal ROOT object hierarchy and metadata | `writer.py` | `Root/SampleHist.cc`, around lines 496–512 | ROOT round-trip and 126-object comparison |
| Compare against authoritative TREx output | `compare.py` | observable contract | paths, axes, titles, contents, and variances |

## Deliberate boundary

The current verifier accepts only fixed-width, one-dimensional, nominal NTUP
histogramming with the input layout and selection/weight structure used by the
Hyy example. Fit and `NormFactor` blocks are reported as downstream operations:
they are consumed later by unmodified TRExFitter. Systematics, alternate input
precedence, Job/Region weights, aliases, friend trees, variable-width bins,
sample arithmetic, smoothing, and unfolding are rejected.

This boundary is useful for SFT creation: an example can be labeled as a
verified atomic operation, an explicitly unsupported operation, or a downstream
TRExFitter operation. “Accepted” never means “probably close.”
