"""Optional ROOT metadata checks; never iterate event arrays or run a fit."""

from dataclasses import replace
from pathlib import Path

from .coffea_backend.config import ConfigError, _parse_blocks, _unquote, parse_config, resolve_sample_files, _host_path
from .coffea_backend.expressions import Expression
from .semantics import items, expand


def inspect_inputs(path: Path, project_dir: Path):
    issues = []
    checked = set()
    def fail(code, message, block=None, setting=None):
        issues.append(dict(code=code, message=message,
                           block=block.kind if block else None,
                           name=block.name if block else None,
                           line=block.line if block else None, setting=setting))
    try:
        import uproot
    except ImportError:
        fail("input_dependency", "Input checks require uproot: uv sync --extra inputs; uv run --extra inputs ...")
        return issues, 0
    try:
        blocks = _parse_blocks(path)
        jobs = [b for b in blocks if b.kind == "Job"]
        if len(jobs) != 1:
            raise ConfigError("Input checks require exactly one Job")
        job = jobs[0]
        mode = _unquote(job.values.get("ReadFrom", "")).upper()
        if mode == "NTUP":
            # Reuse the established Hyy path/weight/selection contract. Fail
            # explicitly rather than claim complete checks for advanced I/O.
            from .coffea_backend.verify import verify_config
            compatibility = verify_config(path)
            io_errors = [i for i in compatibility.issues if i.severity == "error"]
            if io_errors:
                fail("input_coverage", "NTUP metadata checks currently require the supported Coffea NTUP subset: " + "; ".join(i.message for i in io_errors))
                return issues, 0
            config = parse_config(path)
            common = set()
            for region in config.regions:
                common |= Expression(region.selection).names | Expression(region.variable).names
            sample_blocks = {b.name: b for b in blocks if b.kind == "Sample"}
            for sample in config.samples:
                block = sample_blocks[sample.name]
                names = common | Expression(sample.selection).names
                if sample.weight is not None:
                    names |= Expression(sample.weight).names
                names -= {"TRUE", "FALSE"}
                filenames = set()
                # Every pattern must resolve, even if other patterns match.
                for pattern in sample.file_patterns:
                    try:
                        filenames.update(resolve_sample_files(config, replace(sample, file_patterns=(pattern,)), project_dir))
                    except ConfigError as error:
                        fail("missing_input", str(error), block, "NtupleFiles")
                for filename in sorted(filenames):
                    try:
                        with uproot.open(filename) as source:
                            checked.add(str(filename))
                            if config.ntuple_name not in source:
                                fail("missing_tree", f"{filename}: missing tree {config.ntuple_name}", block, "NtupleName")
                                continue
                            tree = source[config.ntuple_name]
                            if not hasattr(tree, "num_entries") or not hasattr(tree, "keys"):
                                fail("not_tree", f"{filename}:{config.ntuple_name} is not a tree", block)
                                continue
                            missing = names - set(tree.keys())
                            if missing:
                                fail("missing_branches", f"{filename}:{config.ntuple_name}: missing {', '.join(sorted(missing))}", block)
                    except (OSError, ValueError, KeyError) as error:
                        fail("unreadable_input", f"{filename}: {error}", block)
        elif mode == "HIST":
            regions = [b for b in blocks if b.kind == "Region"]
            samples = [b for b in blocks if b.kind == "Sample"]
            # Only simple nominal HIST declarations are resolved here. No
            # suffix algebra, systematic variations, friends or overrides.
            for b in blocks:
                advanced = [key for key in b.values if key.startswith("Histo") and key not in {"HistoPath", "HistoFile", "HistoName", "HistoChecks"}]
                if b.kind in {"Systematic", "Unfolding", "Morphing", "EFTConfig"} or advanced:
                    fail("input_coverage", f"Metadata checks do not yet resolve advanced HIST input: {b.kind} {b.name} {advanced}", b)
            if issues:
                return issues, 0
            edges_by_region = {}
            for region in regions:
                for sample in samples:
                    available = {r.name for r in regions}
                    active = expand(items(sample, "Regions", "all"), available) - expand(items(sample, "Exclude"), available)
                    if region.name not in active:
                        continue
                    # Native basic input settings can be supplied in any of
                    # the three blocks. Ambiguous combinations are not guessed.
                    resolved = {}
                    for key in ("HistoPath", "HistoFile", "HistoName"):
                        values = {value for b in (job, region, sample) for value in items(b, key)}
                        if len(values) != 1:
                            fail("input_coverage", f"{region.name}/{sample.name}: require one unambiguous {key}", sample, key)
                            break
                        resolved[key] = values.pop()
                    if len(resolved) != 3:
                        continue
                    base = _host_path(resolved["HistoPath"], project_dir)
                    pattern = resolved["HistoFile"]
                    if not pattern.endswith(".root"):
                        pattern += ".root"
                    files = sorted(base.glob(pattern))
                    if not files:
                        fail("missing_input", f"No files match {base / pattern}", sample, "HistoFile")
                    for filename in files:
                        try:
                            with uproot.open(filename) as source:
                                checked.add(str(filename))
                                name = resolved["HistoName"]
                                if name not in source:
                                    fail("missing_histogram", f"{filename}: missing {name}", sample, "HistoName")
                                    continue
                                histogram = source[name]
                                if not histogram.classname.startswith("TH1"):
                                    fail("not_histogram", f"{filename}:{name} is not TH1", sample)
                                    continue
                                edges = histogram.axis().edges().tolist()
                                previous = edges_by_region.setdefault(region.name, edges)
                                if edges != previous:
                                    fail("histogram_binning", f"{region.name}: inconsistent edges for {filename}:{name}", sample)
                        except (OSError, ValueError, KeyError) as error:
                            fail("unreadable_input", f"{filename}: {error}", sample)
        else:
            fail("input_coverage", f"Input checks do not support ReadFrom={mode}")
    except (ConfigError, OSError, ValueError) as error:
        fail("input_config", str(error))
    return issues, len(checked)
