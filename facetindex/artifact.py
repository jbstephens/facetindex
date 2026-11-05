"""On-disk layout, manifest, and atomic version publishing.

Layout::

    <root>/
      CURRENT                      # tiny JSON pointer: {"version": "...", "published_at": ...}
      versions/
        <version>/
          manifest.json            # row count, columns, kinds, file inventory, format version
          stats.json               # AttributeProfile per attribute (the catalog)
          ids.parquet              # entity id per position (positions are sorted by id)
          postings.parquet         # (attribute, kind, value, cardinality, roaring bitmap bytes)
          numeric_sorted.parquet   # one row group per numeric-capable attribute: (value, position)
          keys/<attr>.parquet      # dense float64 sort key per position (NaN = missing)
          values.parquet           # normalised text of every short attribute (the narrow table)
          signatures.parquet       # optional forward index: id -> per-entity bitset
          dimensions.json          # optional: which bit means which (attribute, value)

Publishing is two atomic renames: the version directory is built under a temp name and
renamed into place, then CURRENT is replaced via write-temp + rename. Readers that opened
an older version keep working; new readers see the new version. On object storage, the
same idea is "write all objects, then overwrite the pointer object".
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

MANIFEST = "manifest.json"
STATS = "stats.json"
IDS = "ids.parquet"
POSTINGS = "postings.parquet"
NUMERIC_SORTED = "numeric_sorted.parquet"
KEYS_DIR = "keys"
VALUES = "values.parquet"
SIGNATURES = "signatures.parquet"
DIMENSIONS = "dimensions.json"
CURRENT = "CURRENT"
VERSIONS_DIR = "versions"


@dataclass
class Manifest:
    version: str
    created_at: str
    format_version: int
    row_count: int
    id_column: str
    source: dict[str, Any]
    columns: list[str]                         # all attributes, source order
    kinds: dict[str, str]                      # attr -> kind
    numeric_row_groups: dict[str, int]         # attr -> row group index in numeric_sorted.parquet
    key_attrs: list[str]                       # attrs with keys/<attr>.parquet
    value_attrs: list[str]                     # attrs present in values.parquet
    files: dict[str, int] = field(default_factory=dict)   # relative path -> bytes
    build_seconds: float = 0.0
    has_signatures: bool = False
    library_version: str = "0.1.0"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        d = json.loads(s)
        d.setdefault("has_signatures", False)
        d.setdefault("library_version", "unknown")
        return cls(**d)


def versions_dir(root: str | os.PathLike) -> Path:
    return Path(root) / VERSIONS_DIR


def version_path(root: str | os.PathLike, version: str) -> Path:
    if not version or "/" in version or version.startswith("."):
        raise ValueError(f"bad version name {version!r}")
    return versions_dir(root) / version


def new_build_dir(root: str | os.PathLike, version: str) -> Path:
    """A temp directory on the same filesystem as the final location (so rename is atomic)."""
    vd = versions_dir(root)
    vd.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".tmp-{version}-", dir=vd))


def _fsync_dir(p: Path) -> None:
    try:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # not all filesystems support directory fsync


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def finalize_version(build_dir: Path, root: str | os.PathLike, version: str, overwrite: bool = False) -> Path:
    """Rename the temp build directory into ``versions/<version>``."""
    final = version_path(root, version)
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"version {version!r} already exists at {final}")
        old = final.with_name(f".old-{version}-{int(time.time())}")
        os.replace(final, old)
        shutil.rmtree(old, ignore_errors=True)
    for p in build_dir.rglob("*"):
        if p.is_file():
            with open(p, "rb") as f:
                os.fsync(f.fileno())
    os.replace(build_dir, final)
    _fsync_dir(final.parent)
    return final


def set_current(root: str | os.PathLike, version: str) -> None:
    """Point CURRENT at ``version``. Atomic; readers see either the old or the new pointer."""
    vp = version_path(root, version)
    if not (vp / MANIFEST).exists():
        raise FileNotFoundError(f"version {version!r} has no manifest at {vp}")
    atomic_write_text(Path(root) / CURRENT, json.dumps({"version": version, "published_at": time.time()}))


def current_version(root: str | os.PathLike) -> str | None:
    p = Path(root) / CURRENT
    try:
        return json.loads(p.read_text())["version"]
    except FileNotFoundError:
        return None


def list_versions(root: str | os.PathLike) -> list[str]:
    vd = versions_dir(root)
    if not vd.exists():
        return []
    out = []
    for p in vd.iterdir():
        if p.is_dir() and not p.name.startswith(".") and (p / MANIFEST).exists():
            out.append(p.name)
    return sorted(out, key=lambda v: (version_path(root, v) / MANIFEST).stat().st_mtime)


def retain(root: str | os.PathLike, keep: int = 2) -> list[str]:
    """Delete all but the newest ``keep`` versions, never the CURRENT one. Returns deleted names.

    Keep at least 2 so a reader that opened the previous version mid-request finishes cleanly.
    """
    keep = max(1, int(keep))
    cur = current_version(root)
    vs = list_versions(root)
    protected = set(vs[-keep:]) | ({cur} if cur else set())
    deleted = []
    for v in vs:
        if v not in protected:
            shutil.rmtree(version_path(root, v), ignore_errors=True)
            deleted.append(v)
    # sweep abandoned temp build dirs older than an hour
    vd = versions_dir(root)
    if vd.exists():
        cutoff = time.time() - 3600
        for p in vd.iterdir():
            if p.name.startswith(".tmp-") and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
    return deleted


def read_manifest(path: str | os.PathLike) -> Manifest:
    return Manifest.from_json((Path(path) / MANIFEST).read_text())


def inventory(path: Path) -> dict[str, int]:
    files = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(path))] = p.stat().st_size
    return files
