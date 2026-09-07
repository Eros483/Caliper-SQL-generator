"""Semantic SQL cache — arch §5.9. hash(normalized q + org_id + mart_version) -> SQL."""

import hashlib
import re


def _normalize(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def cache_key(question: str, org_id: int | None, mart_version: str) -> str:
    n = _normalize(question)
    raw = f"{n}|{org_id}|{mart_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


class SemanticCache:
    def __init__(self, mart_version: str = "v1"):
        self._mart_version = mart_version
        self._store: dict[str, str] = {}

    @property
    def mart_version(self) -> str:
        return self._mart_version

    @mart_version.setter
    def mart_version(self, value: str) -> None:
        if value != self._mart_version:
            self._store.clear()
        self._mart_version = value

    def make_key(self, question: str, org_id: int | None) -> str:
        return cache_key(question, org_id, self.mart_version)

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, sql: str) -> None:
        self._store[key] = sql

    def invalidate(self) -> None:
        self._store.clear()


# global singleton — used by agent; invalidates on nightly dbt run via mart version bump
semantic_cache = SemanticCache()
