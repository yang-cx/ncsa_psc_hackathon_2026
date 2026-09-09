"""Coffea processor that evaluates every configured region in one event pass."""

from __future__ import annotations

import awkward as ak
import numpy as np
from coffea import processor
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from .config import AnalysisConfig
from .expressions import Expression, boolean_mask, dense_values


class TrexNtupleProcessor(processor.ProcessorABC):
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._samples = {sample.name: sample for sample in config.samples}
        self._sample_selections = {
            sample.name: Expression(sample.selection) for sample in config.samples
        }
        self._sample_weights = {
            sample.name: Expression(sample.weight)
            for sample in config.samples
            if sample.weight is not None
        }
        self._region_selections = {
            region.name: Expression(region.selection) for region in config.regions
        }
        self._region_variables = {
            region.name: Expression(region.variable) for region in config.regions
        }
        common_names = set().union(
            *(expression.names for expression in self._region_selections.values()),
            *(expression.names for expression in self._region_variables.values()),
        )
        self._required_names = {}
        for sample in config.samples:
            names = common_names | self._sample_selections[sample.name].names
            if sample.name in self._sample_weights:
                names |= self._sample_weights[sample.name].names
            self._required_names[sample.name] = names - {"TRUE", "FALSE"}

    @staticmethod
    def _histogram(region) -> Hist:
        return Hist(
            Regular(
                region.bins,
                region.minimum,
                region.maximum,
                name=region.name,
                label=region.variable_title,
                underflow=True,
                overflow=True,
            ),
            storage=Weight(),
        )

    @staticmethod
    def _evaluate(expression: Expression, columns):
        # ROOT selections commonly guard calculations that are undefined for
        # rejected events. Vectorized evaluation has no short-circuit, so
        # suppress the expected intermediate NaN/divide warnings; the final
        # finite-value mask still rejects them.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return expression(columns)

    def process(self, events):
        sample_name = events.metadata["dataset"]
        sample = self._samples[sample_name]
        # NanoEvents columns are lazy. Materializing every required branch once
        # avoids a fresh ROOT read for each repeated name in long region
        # expressions, while retaining Coffea's chunked event model.
        columns = {
            name: ak.materialize(Expression.resolve_field(events, name))
            for name in self._required_names[sample_name]
        }
        sample_mask = boolean_mask(
            self._evaluate(self._sample_selections[sample_name], columns), len(events)
        )
        if sample.is_data:
            weights = np.ones(len(events), dtype=np.float64)
        else:
            weights = dense_values(
                self._evaluate(self._sample_weights[sample_name], columns), len(events)
            )
            weights *= self.config.luminosity

        histograms = {}
        for region in self.config.regions:
            histogram = self._histogram(region)
            region_mask = boolean_mask(
                self._evaluate(self._region_selections[region.name], columns),
                len(events),
            )
            values = dense_values(
                self._evaluate(self._region_variables[region.name], columns),
                len(events),
            )
            selected = (
                sample_mask
                & region_mask
                & np.isfinite(values)
                & np.isfinite(weights)
            )
            histogram.fill(values[selected], weight=weights[selected])
            histograms[region.name] = histogram
        return {sample_name: histograms}

    def postprocess(self, accumulator):
        return accumulator
