"""Persistent storage and matching for locally registered face embeddings."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, or zero for zero-length vectors."""
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimensions")
    if not left:
        raise ValueError("Embeddings cannot be empty")

    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


class FaceRegistry:
    """A JSON-backed set of named face embeddings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.people: dict[str, list[list[float]]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read face registry '{self.path}': {error}") from error

        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"Unsupported face registry format in '{self.path}'")
        people = data.get("people")
        if not isinstance(people, dict):
            raise ValueError(f"Face registry '{self.path}' has no valid people map")

        for name, embeddings in people.items():
            if not isinstance(name, str) or not isinstance(embeddings, list):
                raise ValueError(f"Face registry '{self.path}' contains an invalid person entry")
            self.people[name] = [self._validate_embedding(item) for item in embeddings]

    @staticmethod
    def _validate_embedding(embedding: Sequence[float]) -> list[float]:
        if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
            raise ValueError("Embedding must be a non-empty sequence of finite numbers")
        values = [float(value) for value in embedding]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding must be a non-empty sequence of finite numbers")
        return values

    def register(self, name: str, embedding: Sequence[float]) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Person name cannot be empty")
        vector = self._validate_embedding(embedding)
        self.people.setdefault(name, []).append(vector)
        self.save()

    def identify(
        self, embedding: Sequence[float], threshold: float = 0.363
    ) -> tuple[str, float] | None:
        vector = self._validate_embedding(embedding)
        best_match: tuple[str, float] | None = None
        for name, known_embeddings in self.people.items():
            for known in known_embeddings:
                if len(known) != len(vector):
                    continue
                score = cosine_similarity(vector, known)
                if best_match is None or score > best_match[1]:
                    best_match = (name, score)
        if best_match is None or best_match[1] < threshold:
            return None
        return best_match

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "people": self.people}
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary_path.replace(self.path)