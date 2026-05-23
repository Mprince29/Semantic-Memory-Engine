import json

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import SemanticMemoryObject
from semantic_memory.extraction.extractor import HashEmbedder
from semantic_memory.infrastructure.store import SemanticMemoryStore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class QueryAwareRetriever:
    def __init__(self, store: SemanticMemoryStore, config: EngineConfig | None = None):
        self.store = store
        self.config = config or DEFAULT_CONFIG
        self.embedder = self._load_embedder(self.config.embedding_model)

    def _load_embedder(self, model_name: str):
        if SentenceTransformer is None:
            return HashEmbedder()
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return HashEmbedder()

    def retrieve(self, query: str, session_id: str, top_k: int = 10) -> list[SemanticMemoryObject]:
        query_vector = self.embedder.encode(query)
        query_embedding = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
            where={"session_id": session_id},
        )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        smos: list[SemanticMemoryObject] = []

        for metadata, distance in zip(metadatas, distances):
            record = dict(metadata)
            record["embedding"] = json.loads(record["embedding"])
            record["confidence"] = 1.0 - float(distance)
            smos.append(SemanticMemoryObject(**record))

        smos.sort(key=lambda item: item.confidence, reverse=True)
        return self._apply_token_budget(smos)

    def _apply_token_budget(self, smos: list[SemanticMemoryObject]) -> list[SemanticMemoryObject]:
        selected: list[SemanticMemoryObject] = []
        used_tokens = 0
        for smo in smos:
            cost = max(1, int((len(smo.predicate) + len(smo.value) + len(smo.subject) + 3) / 3.5))
            if used_tokens + cost > self.config.token_budget:
                break
            selected.append(smo)
            used_tokens += cost
        return selected
