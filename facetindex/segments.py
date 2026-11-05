"""Saved segments ("use cases"): named filters, possibly composed from other segments.

Segments live outside the index (they are edited between loads), so they are evaluated on
demand against whatever version is current. Composition is just a ``Segment(name)`` node
inside another segment's filter; cycles are rejected at save time and at evaluation time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import filters as F


@dataclass
class SegmentDef:
    name: str
    filter: object
    label: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description,
                "tags": list(self.tags), "filter": F.to_dict(self.filter)}

    @classmethod
    def from_dict(cls, d: dict) -> SegmentDef:
        return cls(d["name"], F.from_dict(d["filter"]), d.get("label", ""), d.get("description", ""),
                   list(d.get("tags", [])))


class SegmentStore:
    """In-memory registry with JSON persistence. Swap the persistence for your admin DB."""

    def __init__(self, defs: dict[str, SegmentDef] | None = None):
        self._defs: dict[str, SegmentDef] = dict(defs or {})

    # -- persistence
    @classmethod
    def load(cls, path: str | Path) -> SegmentStore:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls({d["name"]: SegmentDef.from_dict(d) for d in data})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([d.to_dict() for d in self._defs.values()], indent=1))

    # -- CRUD with validation
    def put(self, seg: SegmentDef) -> None:
        trial = dict(self._defs)
        trial[seg.name] = seg
        _check_cycles(trial)
        self._defs = trial

    def remove(self, name: str) -> None:
        dependents = [d.name for d in self._defs.values() if name in F.segments_referenced(d.filter)]
        if dependents:
            raise ValueError(f"segment {name!r} is referenced by {dependents}")
        self._defs.pop(name, None)

    def get(self, name: str) -> SegmentDef:
        return self._defs[name]

    def names(self) -> list[str]:
        return sorted(self._defs)

    def filters(self) -> dict[str, object]:
        """name -> filter, the shape FacetIndex.evaluate() and filters.to_sql() take."""
        return {n: d.filter for n, d in self._defs.items()}

    def __len__(self) -> int:
        return len(self._defs)

    def __contains__(self, name: str) -> bool:
        return name in self._defs

    def dependencies(self, name: str) -> list[str]:
        """Transitive segment dependencies, in evaluation order."""
        out: list[str] = []
        seen: set[str] = set()

        def walk(n):
            for ref in sorted(F.segments_referenced(self._defs[n].filter)):
                if ref in self._defs and ref not in seen:
                    seen.add(ref)
                    walk(ref)
                    out.append(ref)

        walk(name)
        return out


def _check_cycles(defs: dict[str, SegmentDef]) -> None:
    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(defs, WHITE)

    def visit(n, path):
        color[n] = GREY
        for ref in F.segments_referenced(defs[n].filter):
            if ref not in defs:
                continue  # dangling reference: allowed (evaluates empty), reported by validate()
            if color[ref] == GREY:
                raise ValueError(f"segment cycle: {' -> '.join(path + [n, ref])}")
            if color[ref] == WHITE:
                visit(ref, path + [n])
        color[n] = BLACK

    for n in defs:
        if color[n] == WHITE:
            visit(n, [])
