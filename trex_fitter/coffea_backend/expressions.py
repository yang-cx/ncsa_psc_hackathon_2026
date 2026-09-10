"""Safe vectorized evaluator for the ROOT-style expressions in TREx configs."""

from __future__ import annotations

import ast
import operator
import re
from functools import reduce
from typing import Any, Mapping

import awkward as ak
import numpy as np


class ExpressionError(ValueError):
    """An expression uses syntax outside the supported config subset."""


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS = {
    "abs": np.abs,
    "fabs": np.abs,
    "sqrt": np.sqrt,
    "cosh": np.cosh,
    "cos": np.cos,
    "sin": np.sin,
    "tan": np.tan,
    "acos": np.arccos,
    "asin": np.arcsin,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "exp": np.exp,
    "log": np.log,
    "pow": np.power,
    "min": np.minimum,
    "max": np.maximum,
}


def normalize(expression: str) -> str:
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return expression.strip()


class Expression:
    """A parsed expression that never invokes Python ``eval``."""

    def __init__(self, source: str):
        self.source = source
        try:
            self.tree = ast.parse(normalize(source), mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(f"Invalid expression {source!r}: {exc.msg}") from exc
        self._validate(self.tree)

    @property
    def names(self) -> set[str]:
        return {
            node.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name) and node.id not in _FUNCTIONS
        }

    def __call__(self, events: Any) -> Any:
        return self._evaluate(self.tree, events)

    def _validate(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.USub,
            ast.UAdd,
            ast.BinOp,
            *tuple(_BINARY),
            ast.Compare,
            *tuple(_COMPARE),
            ast.Subscript,
            ast.Load,
            ast.Call,
        )
        for child in ast.walk(node):
            if not isinstance(child, allowed):
                raise ExpressionError(
                    f"Unsupported {type(child).__name__} in expression {self.source!r}"
                )
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in _FUNCTIONS:
                    raise ExpressionError(
                        f"Unsupported function call in expression {self.source!r}"
                    )
                if child.keywords:
                    raise ExpressionError("Keyword arguments are not supported")
                arity = 2 if child.func.id in {"atan2", "pow", "min", "max"} else 1
                if len(child.args) != arity:
                    raise ExpressionError(f"{child.func.id} requires {arity} argument(s)")
            if isinstance(child, ast.Subscript):
                if not (isinstance(child.slice, ast.Constant)
                        and type(child.slice.value) is int and child.slice.value >= 0):
                    raise ExpressionError("Only non-negative constant integer collection indices are supported")
            if isinstance(child, ast.Constant) and type(child.value) not in {int, float, bool}:
                raise ExpressionError("Only numeric and Boolean constants are supported")

    @staticmethod
    def resolve_field(events: Any, name: str) -> Any:
        if name == "TRUE":
            return True
        if name == "FALSE":
            return False
        if isinstance(events, Mapping):
            try:
                return events[name]
            except KeyError as exc:
                raise ExpressionError(
                    f"Materialized input has no expression field {name!r}"
                ) from exc
        fields = set(getattr(events, "fields", []))
        if name in fields:
            return events[name]
        # atlas-schema groups flat ``collection_field`` branches into records.
        if "_" in name:
            collection, field = name.split("_", 1)
            if collection in fields:
                collection_array = events[collection]
                if field in set(getattr(collection_array, "fields", [])):
                    return collection_array[field]
        raise ExpressionError(
            f"Input has no field corresponding to expression name {name!r}"
        )

    def _evaluate(self, node: ast.AST, events: Any) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.resolve_field(events, node.id)
        if isinstance(node, ast.Subscript):
            array = self._evaluate(node.value, events)
            if not isinstance(node.slice, ast.Constant) or not isinstance(
                node.slice.value, int
            ):
                raise ExpressionError("Only constant integer collection indices are supported")
            index = node.slice.value
            if index < 0:
                raise ExpressionError("Negative collection indices are not supported")
            return ak.pad_none(array, index + 1, axis=1)[:, index]
        if isinstance(node, ast.BoolOp):
            values = [self._evaluate(value, events) for value in node.values]
            function = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            # Three-valued logic: a known true operand determines OR, and a
            # known false operand determines AND even if another is missing.
            def combine(left, right):
                if np.isscalar(left) and np.isscalar(right):
                    return function(left, right)
                missing_left = False if np.isscalar(left) else ak.is_none(left, axis=-1)
                missing_right = False if np.isscalar(right) else ak.is_none(right, axis=-1)
                fill = isinstance(node.op, ast.And)
                left = left if np.isscalar(left) else ak.fill_none(left, fill)
                right = right if np.isscalar(right) else ak.fill_none(right, fill)
                result = function(left, right)
                unknown = (missing_left | missing_right) & (result if fill else ~result)
                return ak.mask(result, ~unknown)
            return reduce(combine, values)
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, events)
            if isinstance(node.op, ast.Not):
                return np.logical_not(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            function = _BINARY.get(type(node.op))
            if function is not None:
                return function(
                    self._evaluate(node.left, events),
                    self._evaluate(node.right, events),
                )
        if isinstance(node, ast.Compare):
            left = self._evaluate(node.left, events)
            comparisons = []
            for operation, comparator_node in zip(node.ops, node.comparators):
                right = self._evaluate(comparator_node, events)
                comparisons.append(_COMPARE[type(operation)](left, right))
                left = right
            return reduce(np.logical_and, comparisons)
        if isinstance(node, ast.Call):
            function = _FUNCTIONS[node.func.id]  # validated above
            return function(*(self._evaluate(argument, events) for argument in node.args))
        raise ExpressionError(
            f"Unsupported {type(node).__name__} in expression {self.source!r}"
        )


def boolean_mask(value: Any, size: int | None = None) -> np.ndarray:
    """Convert an option-valued Awkward selection to a dense boolean mask."""
    if np.isscalar(value):
        if size is None:
            raise ExpressionError("A scalar mask requires an event count")
        return np.full(size, bool(value), dtype=bool)
    return ak.to_numpy(ak.fill_none(value, False)).astype(bool, copy=False)


def dense_values(value: Any, size: int | None = None) -> np.ndarray:
    """Convert an option-valued expression to a float array, using NaN for missing."""
    if np.isscalar(value):
        if size is None:
            raise ExpressionError("A scalar value requires an event count")
        return np.full(size, float(value), dtype=np.float64)
    return ak.to_numpy(ak.fill_none(value, np.nan)).astype(float, copy=False)
