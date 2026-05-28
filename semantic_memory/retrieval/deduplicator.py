from typing import Iterable

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import SemanticMemoryObject

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


class SemanticDeduplicator:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG

    def deduplicate(
        self,
        new_smos: list[SemanticMemoryObject],
        existing_smos: list[SemanticMemoryObject],
    ) -> tuple[list[SemanticMemoryObject], list[str]]:
        to_add: list[SemanticMemoryObject] = []
        to_remove: list[str] = []

        for candidate in new_smos:
            match = self._find_match(candidate, existing_smos)
            if match is None:
                to_add.append(candidate)
                continue
            winner = self._pick_winner(candidate, match)
            if winner.id == candidate.id:
                to_add.append(candidate)
                to_remove.append(match.id)

        return to_add, to_remove

    def _find_match(
        self,
        candidate: SemanticMemoryObject,
        existing_smos: Iterable[SemanticMemoryObject],
    ) -> SemanticMemoryObject | None:
        candidate_vector = self._to_vector(candidate.embedding)
        for existing in existing_smos:
            if not self._is_comparable(candidate, existing):
                continue
            existing_vector = self._to_vector(existing.embedding)
            if len(candidate_vector) != len(existing_vector):
                continue
            similarity = self._cosine_similarity(candidate_vector, existing_vector)
            if similarity >= self.config.similarity_threshold:
                return existing
        return None

    @staticmethod
    def _to_vector(values: list[float]):
        if np is not None:
            return np.array(values, dtype=float)
        return [float(value) for value in values]

    @staticmethod
    def _cosine_similarity(left, right) -> float:
        if np is not None:
            denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
            return float(np.dot(left, right) / denom)

        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        denom = (left_norm * right_norm) + 1e-9
        dot = sum(a * b for a, b in zip(left, right))
        return float(dot / denom)

    @staticmethod
    def _pick_winner(left: SemanticMemoryObject, right: SemanticMemoryObject) -> SemanticMemoryObject:
        if len(left.value) != len(right.value):
            return left if len(left.value) > len(right.value) else right
        return left if left.timestamp >= right.timestamp else right

    @staticmethod
    def _is_comparable(left: SemanticMemoryObject, right: SemanticMemoryObject) -> bool:
        if left.type != right.type or left.subject != right.subject:
            return False
        if left.type in {"task", "constraint", "preference", "event", "fact"}:
            return left.predicate == right.predicate
        return True
