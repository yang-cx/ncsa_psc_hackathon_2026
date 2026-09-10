"""Static checks derived from TRExFitter v1.10.0 ConfigReader.cc.

No file access beyond the caller's already-parsed configuration.
"""

import math
from collections import Counter

from .config_format import _unquote, split_top_level


def items(block, key, default=""):
    return split_top_level(block.values.get(key, default))


def expand(names, available):
    if any(name.upper() == "ALL" for name in names):
        return set(available)
    return set(names) - {name for name in names if name.upper() == "NONE"}


def check_semantics(blocks, actions="nwfs"):
    issues = []
    def issue(block, code, message, setting, source):
        issues.append(dict(code=code, message=message, block=block.kind,
                           name=block.name, setting=setting, line=block.line,
                           source_ref=f"TRExFitter-v1.10.0/Root/ConfigReader.cc:{source}"))
    jobs = [b for b in blocks if b.kind == "Job"]
    if len(jobs) != 1:
        return issues
    job = jobs[0]
    samples = [b for b in blocks if b.kind == "Sample"]
    regions = [b for b in blocks if b.kind == "Region"]
    sample_names = {b.name for b in samples}
    region_names = {b.name for b in regions}
    pois = items(job, "POI")
    nfs = [b for b in blocks if b.kind == "NormFactor"]
    declared = {b.name for b in nfs}
    fits = [b for b in blocks if b.kind == "Fit"]
    eft = any(_unquote(b.values.get("FitType", "")).upper() == "EFT" for b in fits)
    split_eft = any(_unquote(b.values.get("SplitSamplesPerBin", "FALSE")).upper() == "TRUE"
                    for b in blocks if b.kind == "EFTConfig")
    for b in blocks:
        if b.kind == "Sample":
            for definition in items(b, "NormFactor")[:1]:
                declared.add(definition)
            template = _unquote(b.values.get("Template", "")).split("@")[-1]
            for pair in split_top_level(template):
                if ":" in pair:
                    declared.add(pair.split(":", 1)[0].strip())
            if eft and not split_eft:
                for pair in items(b, "EFTValue"):
                    if "=" in pair:
                        declared.add(pair.split("=", 1)[0].strip())
        elif b.kind == "ShapeFactor":
            for pair in _unquote(b.values.get("Expression", "")).split("@"):
                if ":" in pair:
                    declared.add(pair.split(":", 1)[0].strip())
        elif b.kind == "Systematic":
            np_name = _unquote(b.values.get("NuisanceParameter", b.name))
            declared.update((np_name, "alpha_" + np_name))
    for poi in pois:
        if poi not in declared:
            issue(job, "unknown_poi", f"POI {poi!r} is not a declared norm factor, nuisance, template, shape-expression or EFT parameter", "POI", 9167)

    if any(a in "flsrix" for a in "".join(actions)) and regions and all(
        _unquote(b.values.get("Type", "SIGNAL")).upper() == "VALIDATION" for b in regions
    ):
        issue(job, "no_fit_region", "Requested fit actions require a non-validation region", "POI", 3343)

    for b in blocks:
        reference_keys = {}
        if b.kind == "Sample":
            reference_keys = {"Regions": region_names, "Exclude": region_names}
        elif b.kind in {"NormFactor", "ShapeFactor", "Systematic"}:
            reference_keys = {"Samples": sample_names, "Regions": region_names,
                              "Exclude": sample_names | region_names}
        if b.kind == "Systematic":
            for key in ("DropShapeIn", "DropNormIn", "DropNorm", "DropNormSpecial", "KeepNormForSamples"):
                reference_keys[key] = sample_names | region_names
        for key, available in reference_keys.items():
            unknown = expand(items(b, key), available) - available
            if unknown:
                code = "unknown_sample" if key == "Samples" else "unknown_region" if key == "Regions" else "unknown_reference"
                issue(b, code, f"{key} references unknown names: {', '.join(sorted(unknown))}", key, 8761)

        if b.kind == "Fit" and "POIAsimov" in b.values:
            values = items(b, "POIAsimov")
            for value in values:
                pair = value.split("@")
                try:
                    if len(pair) == 1 and len(values) == 1 and len(pois) == 1:
                        number = float(pair[0])
                    elif len(pair) == 2 and pair[0].strip() in pois:
                        number = float(pair[1])
                    else:
                        raise ValueError()
                    if not math.isfinite(number):
                        raise ValueError()
                except ValueError:
                    issue(b, "invalid_poi_asimov", "POIAsimov requires a finite scalar for one POI, or declared-name@finite-value pairs", "POIAsimov", 1845)

        ranges = {"RankingNPfraction": (0, 1)} if b.kind == "Job" else {}
        if b.kind == "Fit":
            ranges.update(NumCPU=(1, math.inf), ToysHistoNbins=(2, math.inf))
            if "doLHscan" in b.values or "do2DLHscan" in b.values:
                ranges.update(LHscanSteps=(3, 500), LHscanStepsY=(3, 100))
        for key, (low, high) in ranges.items():
            if key not in b.values:
                continue
            try:
                value = float(_unquote(b.values[key]))
                if not low <= value <= high:
                    issue(b, "numeric_range", f"{key} must lie in [{low}, {high}]", key, 938 if b.kind == "Job" else 1960)
            except ValueError:
                pass  # Native schema validation already reports type errors.
        if b.kind == "Job":
            for key in ("RankingPOIName", "RankingUpperAxisNdivision", "RankingPOIAxisScale"):
                if key in b.values and len(items(b, key)) != len(pois):
                    issue(b, "poi_list_length", f"{key} must contain {len(pois)} entries, one per POI", key, 6263)

    # Duplicate names are allowed in disjoint regions; reject overlapping
    # attachments, including inline Sample NormFactor definitions.
    for sample in samples:
        active_regions = expand(items(sample, "Regions", "all"), region_names) - expand(items(sample, "Exclude"), region_names)
        for region in active_regions:
            attached = items(sample, "NormFactor")[:1]
            for nf in nfs:
                excluded = expand(items(nf, "Exclude"), sample_names | region_names)
                if sample.name in excluded or region in excluded:
                    continue
                if sample.name in expand(items(nf, "Samples", "all"), sample_names) and region in expand(items(nf, "Regions", "all"), region_names):
                    attached.append(nf.name)
            duplicates = [name for name, count in Counter(attached).items() if count > 1]
            if duplicates:
                issue(sample, "duplicate_normfactor_attachment", f"Duplicate norm factors in {region}: {', '.join(duplicates)}", "NormFactor", 9677)
    return issues
