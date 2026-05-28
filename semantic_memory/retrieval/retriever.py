import json

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import MEMORY_STATE_ACTIVE, SemanticMemoryObject
from semantic_memory.extraction.extractor import HashEmbedder
from semantic_memory.infrastructure.store import SemanticMemoryStore
from semantic_memory.retrieval.complexity import ComplexityResult, QueryComplexityAnalyzer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class QueryAwareRetriever:
    def __init__(self, store: SemanticMemoryStore, config: EngineConfig | None = None):
        self.store = store
        self.config = config or DEFAULT_CONFIG
        self.embedder = self._load_embedder(self.config.embedding_model)
        self._complexity = QueryComplexityAnalyzer(config=self.config)

    def _load_embedder(self, model_name: str):
        if SentenceTransformer is None:
            return HashEmbedder()
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return HashEmbedder()

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        user_id: str = "",
    ) -> tuple[list[SemanticMemoryObject], ComplexityResult]:
        """
        Returns (memories, complexity_result).

        Token budget is derived adaptively from query complexity. Only active
        memories are returned; disputed/superseded/contextual ones are skipped.
        """
        complexity = self._complexity.analyze(query)
        query_vector = self.embedder.encode(query)
        query_embedding = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        where_filter: dict = {"session_id": session_id}

        try:
            results = self.store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
                where=where_filter,
            )
        except Exception as exc:
            if not self.store._is_dimension_mismatch(exc):
                raise
            self.store.rebuild_vector_index(len(query_embedding))
            results = self.store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
                where=where_filter,
            )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        smos: list[SemanticMemoryObject] = []

        for metadata, distance in zip(metadatas, distances):
            record = dict(metadata)
            record["embedding"] = json.loads(record["embedding"])
            record["confidence"] = 1.0 - float(distance)
            smo = SemanticMemoryObject(**record)
            # Skip non-active memories so contradictions don't pollute SPL
            if smo.memory_state != MEMORY_STATE_ACTIVE:
                continue
            smos.append(smo)

        # Optionally merge federated memories for this user
        if user_id:
            federated = self.store.fetch_federated(user_id=user_id, exclude_session=session_id)
            smos = self._merge_federated(smos, federated)

        smos.sort(key=lambda item: item.confidence, reverse=True)
        selected = self._apply_token_budget(smos, budget=complexity.token_budget)
        return selected, complexity

    def _apply_token_budget(
        self, smos: list[SemanticMemoryObject], budget: int
    ) -> list[SemanticMemoryObject]:
        selected: list[SemanticMemoryObject] = []
        used_tokens = 0
        for smo in smos:
            cost = max(1, int((len(smo.predicate) + len(smo.value) + len(smo.subject) + 3) / 3.5))
            if used_tokens + cost > budget:
                break
            selected.append(smo)
            used_tokens += cost
        return selected

    @staticmethod
    def _merge_federated(
        local: list[SemanticMemoryObject],
        federated: list[SemanticMemoryObject],
    ) -> list[SemanticMemoryObject]:
        """Add federated memories that aren't already covered by local ones."""
        local_ids = {smo.id for smo in local}
        extras = [f for f in federated if f.id not in local_ids and f.memory_state == MEMORY_STATE_ACTIVE]
        return local + extras
