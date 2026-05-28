"""
Semantic Runtime Engine for LLMs — V2

Pipeline:
  ingest_turns() → extract → contradict-check → deduplicate → persist (+ federate)
  answer()       → retrieve (adaptive budget) → schema-encoded SPL → Ollama
"""
from __future__ import annotations

from collections.abc import Iterable

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.contradiction.detector import ContradictionDetector, ContradictionReport
from semantic_memory.domain.models import (
    MEMORY_STATE_ACTIVE,
    VISIBILITY_PRIVATE,
    SemanticMemoryObject,
)
from semantic_memory.extraction.extractor import SemanticExtractor
from semantic_memory.infrastructure.ollama import OllamaClient
from semantic_memory.infrastructure.store import SemanticMemoryStore
from semantic_memory.prompting.builder import PromptBuilder
from semantic_memory.prompting.spl import SPLEncoder
from semantic_memory.retrieval.complexity import ComplexityResult
from semantic_memory.retrieval.deduplicator import SemanticDeduplicator
from semantic_memory.retrieval.retriever import QueryAwareRetriever


class SemanticRuntimeEngine:
    """
    V2 engine. Drop-in replacement for V1 SemanticMemoryEngine with additional
    keyword arguments for schema selection, user identity, and visibility.
    """

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.extractor = SemanticExtractor(self.config)
        self.store = SemanticMemoryStore(self.config)
        self.deduplicator = SemanticDeduplicator(self.config)
        self.contradiction_detector = ContradictionDetector(config=self.config)
        self.retriever = QueryAwareRetriever(self.store, self.config)
        self.prompt_builder = PromptBuilder()
        self.encoder = SPLEncoder()
        self.ollama = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
        )

    # ------------------------------------------------------------------ ingest

    def ingest_turns(
        self,
        turns: Iterable[str],
        session_id: str,
        user_id: str = "",
        visibility: str = VISIBILITY_PRIVATE,
        owner: str = "",
    ) -> list[ContradictionReport]:
        """
        Ingest conversation turns into the memory store.

        Returns one ContradictionReport per turn so callers can inspect
        what was flagged, superseded, or disputed.
        """
        existing = self.store.fetch_by_session(session_id)
        reports: list[ContradictionReport] = []

        for turn in turns:
            new_smos = self.extractor.extract(turn, session_id=session_id)

            # Stamp V2 provenance fields
            for smo in new_smos:
                smo.user_id = user_id
                smo.visibility = visibility
                smo.owner = owner or user_id
                smo.provenance = session_id

            # Contradiction check against existing active memories
            active_existing = [s for s in existing if s.memory_state == MEMORY_STATE_ACTIVE]
            report = self.contradiction_detector.check_batch(new_smos, active_existing, turn)
            reports.append(report)

            # Apply resolved states from contradiction check
            for result in report.results:
                result.candidate.memory_state = result.state_for_candidate
                if result.conflicting and result.state_for_conflicting != result.conflicting.memory_state:
                    self.store.update_memory_state(result.conflicting.id, result.state_for_conflicting)

            # Semantic deduplication (shape-level, within active memories)
            active_candidates = [r.candidate for r in report.results if r.candidate.memory_state == MEMORY_STATE_ACTIVE]
            to_add, to_remove = self.deduplicator.deduplicate(active_candidates, active_existing)

            # Also persist non-active memories (disputed/contextual/superseded) for audit
            non_active = [r.candidate for r in report.results if r.candidate.memory_state != MEMORY_STATE_ACTIVE]

            if to_remove:
                removed_ids = set(to_remove)
                self.store.delete_many(to_remove)
                existing = [s for s in existing if s.id not in removed_ids]

            all_to_store = to_add + non_active
            self.store.upsert_many(all_to_store)

            # Publish shareable memories to the federation pool
            if user_id:
                self.store.publish_to_federation(all_to_store)

            existing = all_to_store + existing

        return reports

    # ------------------------------------------------------------------ query

    def build_spl(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        schema_name: str | None = None,
        user_id: str = "",
    ) -> tuple[str, ComplexityResult]:
        """
        Returns (spl_string, complexity_result).
        Caller can inspect complexity.tier and complexity.token_budget.
        """
        schema = schema_name or self.config.default_schema
        relevant, complexity = self.retriever.retrieve(
            query=query, session_id=session_id, top_k=top_k, user_id=user_id
        )
        contextual = self.store.fetch_contextual_by_session(session_id)
        spl = self.encoder.encode(relevant, query, schema_name=schema, contextual=contextual)
        return spl, complexity

    def answer(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        schema_name: str | None = None,
        user_id: str = "",
    ) -> dict:
        """
        Returns dict with keys: response, schema, tier, token_budget, signals.
        """
        schema = schema_name or self.config.default_schema
        relevant, complexity = self.retriever.retrieve(
            query=query, session_id=session_id, top_k=top_k, user_id=user_id
        )
        contextual = self.store.fetch_contextual_by_session(session_id)
        prompt = self.prompt_builder.build(
            relevant,
            query,
            schema_name=schema,
            contextual=contextual,
        )
        result = self.ollama.generate(prompt, system=self.prompt_builder.system_prompt)
        return {
            "response": result.get("response", ""),
            "schema": schema,
            "tier": complexity.tier,
            "token_budget": complexity.token_budget,
            "signals": complexity.signals,
        }


    def clear_session(self, session_id: str) -> int:
        """Remove all memories for a session. Useful for dev/test resets."""
        return self.store.clear_session(session_id)


# V1 backward-compatibility alias
SemanticMemoryEngine = SemanticRuntimeEngine
