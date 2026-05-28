"""
Contradiction-aware memory resolution for V2.

Resolution rules (applied in priority order):
1. Scope divergence  → mark both contextual  (e.g. "Docker at work" vs "no Docker here")
2. Polarity conflict → mark older disputed, keep newer active
3. Fact update       → mark older superseded, keep newer active
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from semantic_memory.domain.models import (
    MEMORY_STATE_ACTIVE,
    MEMORY_STATE_CONTEXTUAL,
    MEMORY_STATE_DISPUTED,
    MEMORY_STATE_SUPERSEDED,
    SemanticMemoryObject,
)

# Phrases that suggest a memory is scoped to a context rather than universal
_SCOPE_MARKERS = (
    "at work", "for this project", "in this case", "here", "for now",
    "on this machine", "for production", "for dev", "for testing",
    "in this repo", "for this task", "locally", "on the server",
)


@dataclass
class ContradictionResult:
    """Outcome of a contradiction check between one candidate and the memory store."""
    candidate: SemanticMemoryObject
    conflicting: SemanticMemoryObject | None = None
    resolution: str = "none"          # none | superseded | disputed | contextual
    state_for_candidate: str = MEMORY_STATE_ACTIVE
    state_for_conflicting: str = MEMORY_STATE_ACTIVE
    explanation: str = ""


@dataclass
class ContradictionReport:
    """Aggregated report for all contradictions found in one ingest pass."""
    results: list[ContradictionResult] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return any(r.resolution != "none" for r in self.results)

    def conflicts(self) -> list[ContradictionResult]:
        return [r for r in self.results if r.resolution != "none"]

    def summary(self) -> str:
        total = len(self.conflicts())
        if total == 0:
            return "No contradictions detected."
        lines = [f"{total} contradiction(s) detected:"]
        for r in self.conflicts():
            lines.append(
                f"  [{r.resolution.upper()}] '{r.candidate.predicate}:{r.candidate.value}' "
                f"vs '{r.conflicting.predicate}:{r.conflicting.value}' — {r.explanation}"  # type: ignore[union-attr]
            )
        return "\n".join(lines)


class ContradictionDetector:
    """
    Scans incoming memories against existing ones and assigns memory states.

    Does NOT delete anything — state assignment is the resolution mechanism.
    The engine decides what to persist based on resulting states.
    """

    def __init__(self, config=None):
        # Minimum fraction of overlapping tokens for two values to be
        # considered comparable (avoids false contradiction flags on
        # semantically unrelated values sharing only stopwords).
        self._min_overlap = config.contradiction_min_overlap if config else 0.1

    def check(
        self,
        candidate: SemanticMemoryObject,
        existing: list[SemanticMemoryObject],
        source_text: str = "",
    ) -> ContradictionResult:
        for existing_smo in existing:
            if not self._is_comparable_pair(candidate, existing_smo):
                continue

            conflict_type = self._classify_conflict(candidate, existing_smo)
            if conflict_type == "none":
                continue

            scoped = self._is_scoped(source_text)

            if scoped:
                return ContradictionResult(
                    candidate=candidate,
                    conflicting=existing_smo,
                    resolution="contextual",
                    state_for_candidate=MEMORY_STATE_CONTEXTUAL,
                    state_for_conflicting=MEMORY_STATE_CONTEXTUAL,
                    explanation=(
                        f"scope marker detected in source text; both memories may be valid in different contexts"
                    ),
                )

            if conflict_type == "polarity":
                # Newer wins, older becomes disputed
                winner, loser = self._by_recency(candidate, existing_smo)
                return ContradictionResult(
                    candidate=candidate,
                    conflicting=existing_smo,
                    resolution="disputed",
                    state_for_candidate=MEMORY_STATE_ACTIVE if winner is candidate else MEMORY_STATE_DISPUTED,
                    state_for_conflicting=MEMORY_STATE_ACTIVE if winner is existing_smo else MEMORY_STATE_DISPUTED,
                    explanation=(
                        f"direct polarity conflict on '{candidate.predicate}' — "
                        f"newer memory ('{winner.value}') wins"
                    ),
                )

            if conflict_type == "fact_update":
                # Newer fact supersedes older one
                winner, loser = self._by_recency(candidate, existing_smo)
                return ContradictionResult(
                    candidate=candidate,
                    conflicting=existing_smo,
                    resolution="superseded",
                    state_for_candidate=MEMORY_STATE_ACTIVE if winner is candidate else MEMORY_STATE_SUPERSEDED,
                    state_for_conflicting=MEMORY_STATE_SUPERSEDED if winner is candidate else MEMORY_STATE_ACTIVE,
                    explanation=(
                        f"fact update on predicate '{candidate.predicate}': "
                        f"'{loser.value}' → '{winner.value}'"
                    ),
                )

        return ContradictionResult(candidate=candidate)

    def check_batch(
        self,
        candidates: list[SemanticMemoryObject],
        existing: list[SemanticMemoryObject],
        source_text: str = "",
    ) -> ContradictionReport:
        report = ContradictionReport()
        all_known = list(existing)
        for candidate in candidates:
            result = self.check(candidate, all_known, source_text)
            report.results.append(result)
            # Add resolved candidate to known set so intra-batch contradictions are caught
            resolved = SemanticMemoryObject(**{
                f.name: getattr(candidate, f.name) for f in candidate.__dataclass_fields__.values()  # type: ignore[attr-defined]
            })
            resolved.memory_state = result.state_for_candidate
            all_known.append(resolved)
        return report

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _is_comparable_pair(a: SemanticMemoryObject, b: SemanticMemoryObject) -> bool:
        """Two memories are comparable if they share type and predicate family."""
        if a.id == b.id or a.type != b.type:
            return False
        # Preferences with opposite polarity ARE comparable (that's how we detect flips)
        if a.type == "preference":
            return _strip_polarity_prefix(a.predicate) == _strip_polarity_prefix(b.predicate)
        return a.predicate == b.predicate

    def _classify_conflict(
        self,
        candidate: SemanticMemoryObject,
        existing: SemanticMemoryObject,
    ) -> str:
        """Return conflict type: 'polarity' | 'fact_update' | 'none'."""
        if candidate.type == "preference":
            c_core = _strip_polarity_prefix(candidate.predicate)
            e_core = _strip_polarity_prefix(existing.predicate)
            if c_core == e_core and candidate.predicate != existing.predicate:
                if _values_overlap(candidate.value, existing.value, self._min_overlap):
                    return "polarity"

        if candidate.type in {"fact", "constraint"}:
            if candidate.value.lower() != existing.value.lower():
                return "fact_update"

        return "none"

    @staticmethod
    def _is_scoped(text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in _SCOPE_MARKERS)

    @staticmethod
    def _by_recency(
        a: SemanticMemoryObject, b: SemanticMemoryObject
    ) -> tuple[SemanticMemoryObject, SemanticMemoryObject]:
        """Return (winner, loser) by timestamp; tie goes to candidate."""
        return (a, b) if a.timestamp >= b.timestamp else (b, a)


def _strip_polarity_prefix(predicate: str) -> str:
    return re.sub(r"^pref_(pos|neg)$", "pref", predicate)


def _values_overlap(a: str, b: str, min_overlap: float = 0.1) -> bool:
    """True when shared token fraction meets the min_overlap threshold."""
    tokens_a = set(a.lower().replace("_", " ").split())
    tokens_b = set(b.lower().replace("_", " ").split())
    stopwords = {"a", "an", "the", "to", "of", "in", "for", "and", "or", "it"}
    tokens_a -= stopwords
    tokens_b -= stopwords
    if not tokens_a or not tokens_b:
        return False
    shared = tokens_a & tokens_b
    # Jaccard-like: shared / union, must meet min_overlap floor
    ratio = len(shared) / len(tokens_a | tokens_b)
    return ratio >= min_overlap
