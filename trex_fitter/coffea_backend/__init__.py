"""Coffea implementation of the TRExFitter NTUP (``n``) action."""

# Keep package import lightweight so the static verifier can run without
# importing Coffea, Uproot, or opening any analysis inputs.
__all__ = ["RunSummary", "run_histogramming"]


def __getattr__(name):
    if name in __all__:
        from .backend import RunSummary, run_histogramming

        return {"RunSummary": RunSummary, "run_histogramming": run_histogramming}[name]
    raise AttributeError(name)
