"""Per-process version follower: hand out the FacetIndex for the CURRENT version.

Each API worker holds one Catalog. On every request it reads the tiny CURRENT pointer
(one small file read, microseconds) and, if the version changed, opens the new artifact
lazily *in the request*. No threads, no cross-process locks: the artifact is immutable and
the pointer flip is atomic, so a worker can never observe a half-written version.

The previous version stays loaded until ``keep`` newer ones exist, so a request that
started on version N finishes on version N even if N+1 is published mid-flight.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import artifact as A
from .index import FacetIndex


class Catalog:
    def __init__(self, root: str | Path, keep: int = 2, poll_seconds: float = 0.0, warm: bool = False):
        self.root = Path(root)
        self.keep = max(1, keep)
        self.poll_seconds = poll_seconds   # 0 = check the pointer on every call
        self.warm = warm
        self._loaded: dict[str, FacetIndex] = {}
        self._order: list[str] = []
        self._last_check = 0.0
        self._pointer_mtime_ns = -1
        self._current_version: str | None = None

    def _read_pointer(self) -> str | None:
        p = self.root / A.CURRENT
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return None
        if st.st_mtime_ns == self._pointer_mtime_ns and self._current_version:
            return self._current_version
        self._pointer_mtime_ns = st.st_mtime_ns
        self._current_version = json.loads(p.read_text())["version"]
        return self._current_version

    def current_version(self) -> str | None:
        now = time.monotonic()
        if self.poll_seconds and now - self._last_check < self.poll_seconds and self._current_version:
            return self._current_version
        self._last_check = now
        return self._read_pointer()

    def get(self, version: str | None = None) -> FacetIndex:
        """The index for CURRENT (or an explicit version). Pin it for the whole request."""
        v = version or self.current_version()
        if v is None:
            raise FileNotFoundError(f"no CURRENT version under {self.root}")
        idx = self._loaded.get(v)
        if idx is None:
            idx = FacetIndex.open(self.root, v)
            if self.warm:
                idx.warm()
            self._loaded[v] = idx
            self._order.append(v)
            while len(self._order) > self.keep:
                old = self._order.pop(0)
                self._loaded.pop(old, None)
        return idx

    def loaded_versions(self) -> list[str]:
        return list(self._order)
