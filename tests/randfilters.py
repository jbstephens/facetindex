"""Random filter generator shared by the equivalence tests."""

from __future__ import annotations

import random

from facetindex import filters as F


class Gen:
    def __init__(self, stats: dict, seed: int):
        self.rng = random.Random(seed)
        self.stats = stats
        self.by_kind: dict[str, list[str]] = {}
        for name, s in stats.items():
            self.by_kind.setdefault(s["kind"], []).append(name)

    def pick(self, kinds):
        pool = [a for k in kinds for a in self.by_kind.get(k, [])]
        return self.rng.choice(pool) if pool else None

    def leaf(self):
        r = self.rng.random()
        if r < 0.22 and self.by_kind.get("flag"):
            return F.Flag(self.pick(["flag"]), self.rng.random() < 0.7)
        if r < 0.45:
            a = self.pick(["categorical", "numeric", "short_text"])
            top = [v for v, _ in self.stats[a]["top_values"]] or ["x"]
            k = self.rng.randint(1, min(3, len(top)))
            vals = self.rng.sample(top, k)
            if self.rng.random() < 0.3:
                vals.append("no-such-value")
            if self.rng.random() < 0.2:
                vals = [" " + vals[0] + " "]  # whitespace should be trimmed on both sides
            return F.In(a, vals)
        if r < 0.65:
            a = self.pick(["numeric", "short_text"])
            s = self.stats[a]
            lo, hi = s["min"], s["max"]
            if lo is None:
                lo, hi = 0.0, 100.0
            x, y = sorted([self.rng.uniform(lo, hi), self.rng.uniform(lo, hi)])
            if self.rng.random() < 0.2:
                x = None
            if self.rng.random() < 0.2:
                y = None
            return F.Between(a, x, y)
        if r < 0.80:
            a = self.pick(["short_text", "categorical"])
            sample = self.stats[a]["sample"] or "ab"
            i = self.rng.randrange(len(sample))
            needle = sample[i : i + self.rng.randint(1, 3)] or "a"
            if self.rng.random() < 0.3:
                needle = needle.upper()
            if self.rng.random() < 0.1:
                needle = "%"  # must be escaped in the SQL path
            return F.Contains(a, needle)
        a = self.pick(["flag", "categorical", "numeric", "short_text", "long_text", "empty"])
        return F.IsEmpty(a) if self.rng.random() < 0.5 else F.IsNotEmpty(a)

    def tree(self, depth=0, segments: list[str] | None = None):
        r = self.rng.random()
        if depth >= 2 or r < 0.35:
            if segments and self.rng.random() < 0.15:
                return F.Segment(self.rng.choice(segments))
            return self.leaf()
        n = self.rng.randint(1, 4)
        kids = [self.tree(depth + 1, segments) for _ in range(n)]
        if r < 0.75:
            return F.And(kids)
        if r < 0.9:
            return F.Or(kids)
        return F.Not(kids[0])
