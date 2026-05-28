"""
Query complexity analysis for adaptive token budget.

Tiers and their token budgets:
  simple     → 30-50   factual lookups, single-concept questions
  preference → 60-90   questions involving user preferences or task context
  planning   → 100-160 multi-step, comparison, debug, architectural questions
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Keywords that signal each tier
_PLANNING_SIGNALS = frozenset({
    "compare", "comparison", "difference", "tradeoff", "trade-off", "vs",
    "versus", "explain", "why", "how should", "architecture", "design",
    "plan", "strategy", "approach", "evaluate", "pros", "cons",
    "debug", "troubleshoot", "diagnose", "step by step", "migrate",
    "refactor", "optimize", "all", "list all", "every", "what are the",
    "alternatives", "options", "recommend", "best way",
})

_PREFERENCE_SIGNALS = frozenset({
    "prefer", "like", "want", "use", "need", "should i", "which",
    "my setup", "my project", "given my", "based on", "for my",
    "what do you think", "is it better", "recommend",
})

_TEMPORAL_SIGNALS = frozenset({
    "deadline", "tomorrow", "today", "next week", "by when",
    "until", "schedule", "sprint", "release", "ship",
})

# Verb count threshold that escalates complexity
_VERB_ESCALATE_THRESHOLD = 3


@dataclass(frozen=True)
class ComplexityResult:
    tier: str           # simple | preference | planning
    token_budget: int
    score: float        # raw score for debugging / logging
    signals: list[str]  # which signals fired


class QueryComplexityAnalyzer:
    """
    Classifies a query into a complexity tier and returns a token budget.

    Uses only cheap string signals — no model calls, no embeddings.
    Budget values default to the class-level constants but can be overridden
    via EngineConfig (budget_simple, budget_preference, budget_planning).
    """

    TIER_BUDGETS: dict[str, int] = {
        "simple": 45,
        "preference": 75,
        "planning": 140,
    }

    def __init__(self, config=None):
        if config is not None:
            self.TIER_BUDGETS = {
                "simple": config.budget_simple or self.TIER_BUDGETS["simple"],
                "preference": config.budget_preference or self.TIER_BUDGETS["preference"],
                "planning": config.budget_planning or self.TIER_BUDGETS["planning"],
            }

    def analyze(self, query: str) -> ComplexityResult:
        lower = query.lower().strip()
        score = 0.0
        fired: list[str] = []

        # Signal 1: raw length
        word_count = len(lower.split())
        if word_count > 30:
            score += 2.0
            fired.append(f"long_query({word_count}w)")
        elif word_count > 15:
            score += 1.0
            fired.append(f"medium_query({word_count}w)")

        # Signal 2: planning keywords
        for kw in _PLANNING_SIGNALS:
            if kw in lower:
                score += 1.5
                fired.append(f"planning:{kw}")
                break  # one match is enough to escalate

        # Signal 3: preference keywords
        for kw in _PREFERENCE_SIGNALS:
            if kw in lower:
                score += 1.0
                fired.append(f"preference:{kw}")
                break

        # Signal 4: temporal terms (add context weight)
        for kw in _TEMPORAL_SIGNALS:
            if kw in lower:
                score += 0.5
                fired.append(f"temporal:{kw}")
                break

        # Signal 5: multiple question marks or "and"
        if lower.count("?") > 1:
            score += 1.0
            fired.append("multi_question")
        if re.search(r"\band\b.*\band\b", lower):
            score += 0.7
            fired.append("compound_and")

        # Signal 6: verb density (rough — count common planning verbs)
        verb_hits = sum(
            1 for v in ("should", "would", "could", "can", "will", "need", "want", "have to")
            if re.search(rf"\b{v}\b", lower)
        )
        if verb_hits >= _VERB_ESCALATE_THRESHOLD:
            score += 1.0
            fired.append(f"high_verb_density({verb_hits})")

        tier = self._score_to_tier(score)
        return ComplexityResult(
            tier=tier,
            token_budget=self.TIER_BUDGETS[tier],
            score=round(score, 2),
            signals=fired,
        )

    @staticmethod
    def _score_to_tier(score: float) -> str:
        if score >= 2.5:
            return "planning"
        if score >= 1.0:
            return "preference"
        return "simple"
