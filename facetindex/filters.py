"""Filter AST: the one query language shared by the bitmap engine and the SQL emitter.

A filter is a small tree. The same tree can be:
  * evaluated against a FacetIndex (bitmaps, microseconds), or
  * rendered to SQL for a Postgres/DuckDB fallback (``to_sql``), or
  * serialised to/from JSON for saved segments (``to_dict`` / ``from_dict``).

Semantics (pinned down here so both evaluators agree, see docs/DESIGN.md):
  * A value is *empty* if it is NULL or blank after trimming whitespace.
  * ``Flag(attr, True)`` matches truthy spellings (true/t/1/yes/y, case-insensitive);
    ``Flag(attr, False)`` matches falsy spellings (false/f/0/no/n). Empty matches neither.
  * ``In`` compares the trimmed stored text with the trimmed query values exactly (case-sensitive).
  * ``Between`` is inclusive on both ends; values that do not parse as numbers never match.
  * ``Contains`` is a case-insensitive substring match.
  * ``Segment(name)`` references a saved segment by name; it is resolved at evaluation time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Union

TRUTHY = ("true", "t", "1", "yes", "y")
FALSY = ("false", "f", "0", "no", "n")

# Postgres has no TRY_CAST; this regex guards the cast so a stray "n/a" cannot raise.
NUMERIC_RE = r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$"

Filter = Union[
    "And", "Or", "Not", "Flag", "In", "Eq", "Between", "Contains", "IsEmpty", "IsNotEmpty", "Segment", "All"
]


@dataclass(frozen=True)
class All:
    """Matches every entity. Useful as the root of an empty filter."""


@dataclass(frozen=True)
class And:
    children: tuple = ()

    def __init__(self, *children):
        # allow And(a, b) and And([a, b])
        if len(children) == 1 and isinstance(children[0], (list, tuple)):
            children = tuple(children[0])
        object.__setattr__(self, "children", tuple(children))


@dataclass(frozen=True)
class Or:
    children: tuple = ()

    def __init__(self, *children):
        if len(children) == 1 and isinstance(children[0], (list, tuple)):
            children = tuple(children[0])
        object.__setattr__(self, "children", tuple(children))


@dataclass(frozen=True)
class Not:
    child: Any


@dataclass(frozen=True)
class Flag:
    attr: str
    value: bool = True


@dataclass(frozen=True)
class In:
    attr: str
    values: tuple = ()

    def __init__(self, attr: str, values):
        object.__setattr__(self, "attr", attr)
        object.__setattr__(self, "values", tuple(str(v).strip() for v in values))


def Eq(attr: str, value) -> In:  # noqa: N802 - reads like a node
    """Single-value convenience; an ``In`` with one value."""
    return In(attr, [value])


@dataclass(frozen=True)
class Between:
    attr: str
    lo: float | None = None
    hi: float | None = None


@dataclass(frozen=True)
class Contains:
    attr: str
    needle: str


@dataclass(frozen=True)
class IsEmpty:
    attr: str


@dataclass(frozen=True)
class IsNotEmpty:
    attr: str


@dataclass(frozen=True)
class Segment:
    name: str


# ----------------------------------------------------------------------------- JSON codec

_OPS = {
    "all": All,
    "and": And,
    "or": Or,
    "not": Not,
    "flag": Flag,
    "in": In,
    "eq": In,
    "between": Between,
    "contains": Contains,
    "is_empty": IsEmpty,
    "is_not_empty": IsNotEmpty,
    "segment": Segment,
}


def to_dict(f) -> dict:
    """Serialise a filter to a JSON-friendly dict. Stable, so it can be a saved segment."""
    if isinstance(f, All):
        return {"op": "all"}
    if isinstance(f, And):
        return {"op": "and", "children": [to_dict(c) for c in f.children]}
    if isinstance(f, Or):
        return {"op": "or", "children": [to_dict(c) for c in f.children]}
    if isinstance(f, Not):
        return {"op": "not", "child": to_dict(f.child)}
    if isinstance(f, Flag):
        return {"op": "flag", "attr": f.attr, "value": bool(f.value)}
    if isinstance(f, In):
        return {"op": "in", "attr": f.attr, "values": list(f.values)}
    if isinstance(f, Between):
        return {"op": "between", "attr": f.attr, "lo": f.lo, "hi": f.hi}
    if isinstance(f, Contains):
        return {"op": "contains", "attr": f.attr, "needle": f.needle}
    if isinstance(f, IsEmpty):
        return {"op": "is_empty", "attr": f.attr}
    if isinstance(f, IsNotEmpty):
        return {"op": "is_not_empty", "attr": f.attr}
    if isinstance(f, Segment):
        return {"op": "segment", "name": f.name}
    raise TypeError(f"not a filter node: {f!r}")


def from_dict(d: dict):
    """Parse a filter from its dict form. Raises ValueError on malformed input."""
    if not isinstance(d, dict) or "op" not in d:
        raise ValueError(f"filter node must be a dict with an 'op': {d!r}")
    op = d["op"]
    if op not in _OPS:
        raise ValueError(f"unknown filter op {op!r}")
    if op == "all":
        return All()
    if op in ("and", "or"):
        children = [from_dict(c) for c in d.get("children", [])]
        return And(children) if op == "and" else Or(children)
    if op == "not":
        return Not(from_dict(d["child"]))
    if op == "flag":
        return Flag(d["attr"], bool(d.get("value", True)))
    if op == "in":
        return In(d["attr"], d.get("values", []))
    if op == "eq":
        return In(d["attr"], [d["value"]])
    if op == "between":
        return Between(d["attr"], d.get("lo"), d.get("hi"))
    if op == "contains":
        return Contains(d["attr"], str(d["needle"]))
    if op == "is_empty":
        return IsEmpty(d["attr"])
    if op == "is_not_empty":
        return IsNotEmpty(d["attr"])
    if op == "segment":
        return Segment(d["name"])
    raise AssertionError(op)


# ----------------------------------------------------------------------------- helpers

def attributes(f) -> set[str]:
    """All attribute names referenced by a filter (segments are opaque here)."""
    out: set[str] = set()

    def walk(n):
        if isinstance(n, (And, Or)):
            for c in n.children:
                walk(c)
        elif isinstance(n, Not):
            walk(n.child)
        elif isinstance(n, (Flag, In, Between, Contains, IsEmpty, IsNotEmpty)):
            out.add(n.attr)

    walk(f)
    return out


def segments_referenced(f) -> set[str]:
    out: set[str] = set()

    def walk(n):
        if isinstance(n, (And, Or)):
            for c in n.children:
                walk(c)
        elif isinstance(n, Not):
            walk(n.child)
        elif isinstance(n, Segment):
            out.add(n.name)

    walk(f)
    return out


def without_attribute(f, attr: str):
    """Return the filter with every top-level AND clause on ``attr`` removed.

    This is the standard faceted-search trick: when counting the values of facet X,
    ignore the user's current selection on X so the other values stay clickable.
    Only clauses directly under the root (or root AND) are removed; nested logic is kept.
    """
    if isinstance(f, And):
        kept = [c for c in f.children if attributes(c) != {attr}]
        return And(kept) if kept else All()
    if attributes(f) == {attr}:
        return All()
    return f


# ----------------------------------------------------------------------------- SQL emitter

@dataclass
class SQLDialect:
    """How to render the primitive predicates for a given database.

    Columns are assumed to be stored as text (that is the problem statement). If your
    serving table has typed columns, pass ``typed=True`` and the casts are dropped.
    """

    name: str = "postgres"
    placeholder: str = "%s"  # psycopg style; use "?" for duckdb/sqlite
    typed: bool = False

    def q(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def nonempty(self, col: str) -> str:
        if self.typed:
            return f"{col} IS NOT NULL"
        return f"NULLIF(TRIM({col}), '') IS NOT NULL"

    def trimmed(self, col: str) -> str:
        return col if self.typed else f"TRIM({col})"

    def numeric(self, col: str) -> str:
        if self.typed:
            return col
        if self.name == "duckdb":
            return f"TRY_CAST(NULLIF(TRIM({col}), '') AS DOUBLE)"
        # Postgres: guard the cast with a regex so garbage never raises.
        return f"(CASE WHEN {col} ~ '{NUMERIC_RE}' THEN {col}::double precision END)"

    def lower(self, expr: str) -> str:
        return f"LOWER({expr})"


POSTGRES = SQLDialect("postgres", "%s")
DUCKDB = SQLDialect("duckdb", "?")


def to_sql(f, dialect: SQLDialect = POSTGRES, segments: dict | None = None) -> tuple[str, list]:
    """Render a filter as a parameterised SQL boolean expression.

    Returns ``(sql, params)``. Saved segments are inlined from ``segments`` (name -> filter).
    Attribute names are quoted identifiers; values are always bound parameters.
    """
    params: list = []
    seen: list[str] = []

    def ph(v) -> str:
        params.append(v)
        return dialect.placeholder

    def render(n) -> str:
        if isinstance(n, All):
            return "TRUE"
        if isinstance(n, And):
            if not n.children:
                return "TRUE"
            return "(" + " AND ".join(render(c) for c in n.children) + ")"
        if isinstance(n, Or):
            if not n.children:
                return "FALSE"
            return "(" + " OR ".join(render(c) for c in n.children) + ")"
        if isinstance(n, Not):
            # COALESCE so NULL comparisons behave like "does not match" rather than UNKNOWN.
            return f"(NOT COALESCE({render(n.child)}, FALSE))"
        col = dialect.q(n.attr) if hasattr(n, "attr") else None
        if isinstance(n, Flag):
            vocab = TRUTHY if n.value else FALSY
            lst = ", ".join(ph(v) for v in vocab)
            return f"{dialect.lower(dialect.trimmed(col))} IN ({lst})"
        if isinstance(n, In):
            if not n.values:
                return "FALSE"
            lst = ", ".join(ph(v) for v in n.values)
            return f"{dialect.trimmed(col)} IN ({lst})"
        if isinstance(n, Between):
            num = dialect.numeric(col)
            parts = []
            if n.lo is not None:
                parts.append(f"{num} >= {ph(float(n.lo))}")
            if n.hi is not None:
                parts.append(f"{num} <= {ph(float(n.hi))}")
            if not parts:
                return f"{num} IS NOT NULL"
            return "(" + " AND ".join(parts) + ")"
        if isinstance(n, Contains):
            needle = n.needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pat = ph("%" + needle + "%")
            if dialect.name == "duckdb":
                return f"{col} ILIKE {pat} ESCAPE '\\'"
            return f"{col} ILIKE {pat} ESCAPE '\\'"
        if isinstance(n, IsEmpty):
            return f"(NOT {dialect.nonempty(col)})"
        if isinstance(n, IsNotEmpty):
            return dialect.nonempty(col)
        if isinstance(n, Segment):
            if segments is None or n.name not in segments:
                raise KeyError(f"unknown segment {n.name!r}")
            if n.name in seen:
                raise ValueError(f"segment cycle through {n.name!r}")
            seen.append(n.name)
            try:
                return "(" + render(segments[n.name]) + ")"
            finally:
                seen.pop()
        raise TypeError(f"not a filter node: {n!r}")

    return render(f), params


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_safe_identifier(name: str) -> bool:
    return bool(_IDENT_RE.match(name))
