from typing import Iterable

import numpy as np

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import SemanticMemoryObject


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
        candidate_vector = np.array(candidate.embedding, dtype=float)
        for existing in existing_smos:
            if not self._is_comparable(candidate, existing):
                continue
            existing_vector = np.array(existing.embedding, dtype=float)
            similarity = self._cosine_similarity(candidate_vector, existing_vector)
            if similarity >= self.config.similarity_threshold:
                return existing
        return None

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
        return float(np.dot(left, right) / denom)

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
