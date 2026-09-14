"""Conservative equivalence checks for supported TRExFitter expressions.

The normal form proves a small set of identities without executing expressions:
redundant parentheses, reversed comparisons, comparison chains, and reordering
of associative/commutative Boolean, addition, and multiplication operands.
Unknown transformations remain unequal rather than being guessed equivalent.
"""

from __future__ import annotations

import ast
from typing import Any

from .coffea_backend.expressions import Expression
from .config_format import _unquote, split_top_level


Canonical = tuple[Any, ...]


def _sort(items: list[Canonical]) -> tuple[Canonical, ...]:
    return tuple(sorted(items, key=repr))


def _flatten(node: ast.AST, node_type: type[ast.AST], operator_type: type[ast.AST]) -> list[ast.AST]:
    if isinstance(node, node_type) and isinstance(node.op, operator_type):
        if isinstance(node, ast.BoolOp):
            return [child for value in node.values for child in _flatten(value, node_type, operator_type)]
        if isinstance(node, ast.BinOp):
            return [
                *_flatten(node.left, node_type, operator_type),
                *_flatten(node.right, node_type, operator_type),
            ]
    return [node]


def _comparison(left: Canonical, operation: ast.cmpop, right: Canonical) -> Canonical:
    if isinstance(operation, ast.Gt):
        return ("compare", "lt", right, left)
    if isinstance(operation, ast.GtE):
        return ("compare", "le", right, left)
    if isinstance(operation, ast.Lt):
        return ("compare", "lt", left, right)
    if isinstance(operation, ast.LtE):
        return ("compare", "le", left, right)
    if isinstance(operation, (ast.Eq, ast.NotEq)):
        ordered = _sort([left, right])
        return ("compare", "eq" if isinstance(operation, ast.Eq) else "ne", *ordered)
    raise TypeError(f"unsupported comparison: {type(operation).__name__}")


def _canonical(node: ast.AST) -> Canonical:
    if isinstance(node, ast.Constant):
        return ("constant", node.value)
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Subscript):
        return ("index", _canonical(node.value), _canonical(node.slice))
    if isinstance(node, ast.Call):
        return ("call", node.func.id, *(_canonical(argument) for argument in node.args))
    if isinstance(node, ast.UnaryOp):
        operation = {ast.Not: "not", ast.USub: "negative", ast.UAdd: "positive"}[type(node.op)]
        return ("unary", operation, _canonical(node.operand))
    if isinstance(node, ast.BoolOp):
        operation_type = type(node.op)
        values = _flatten(node, ast.BoolOp, operation_type)
        operation = "and" if isinstance(node.op, ast.And) else "or"
        return ("boolean", operation, *_sort([_canonical(value) for value in values]))
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Mult)):
            values = _flatten(node, ast.BinOp, type(node.op))
            operation = "add" if isinstance(node.op, ast.Add) else "multiply"
            return ("binary", operation, *_sort([_canonical(value) for value in values]))
        operation = {
            ast.Sub: "subtract", ast.Div: "divide", ast.Mod: "modulo", ast.Pow: "power"
        }[type(node.op)]
        return ("binary", operation, _canonical(node.left), _canonical(node.right))
    if isinstance(node, ast.Compare):
        left = _canonical(node.left)
        comparisons: list[Canonical] = []
        for operation, comparator in zip(node.ops, node.comparators):
            right = _canonical(comparator)
            comparisons.append(_comparison(left, operation, right))
            left = right
        if len(comparisons) == 1:
            return comparisons[0]
        return ("boolean", "and", *_sort(comparisons))
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def expression_normal_form(source: str) -> Canonical:
    """Return a conservative canonical form for a supported expression."""
    return _canonical(Expression(_unquote(source)).tree)


def expressions_equivalent(left: str, right: str) -> bool:
    """Return true only when the implemented normal form proves equivalence."""
    try:
        return expression_normal_form(left) == expression_normal_form(right)
    except (KeyError, TypeError, ValueError):
        return False


def contract_values_equivalent(setting: str, expected: str, actual: str | None) -> bool:
    """Compare a config value with setting-aware, conservative semantics."""
    if actual is None:
        return False
    if expected == actual:
        return True
    if setting in {"Selection", "MCweight"}:
        return expressions_equivalent(expected, actual)
    if setting == "Variable":
        expected_parts = split_top_level(expected)
        actual_parts = split_top_level(actual)
        if len(expected_parts) != 4 or len(actual_parts) != 4:
            return False
        try:
            same_axis = int(expected_parts[1]) == int(actual_parts[1]) and all(
                float(expected_parts[index]) == float(actual_parts[index]) for index in (2, 3)
            )
        except ValueError:
            return False
        return same_axis and expressions_equivalent(expected_parts[0], actual_parts[0])
    return False
