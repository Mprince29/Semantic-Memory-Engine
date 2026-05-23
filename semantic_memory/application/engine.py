from collections.abc import Iterable

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.extraction.extractor import SemanticExtractor
from semantic_memory.infrastructure.ollama import OllamaClient
from semantic_memory.infrastructure.store import SemanticMemoryStore
from semantic_memory.prompting.builder import PromptBuilder
from semantic_memory.prompting.spl import SPLEncoder
from semantic_memory.retrieval.deduplicator import SemanticDeduplicator
from semantic_memory.retrieval.retriever import QueryAwareRetriever


class SemanticMemoryEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.extractor = SemanticExtractor(self.config)
        self.store = SemanticMemoryStore(self.config)
        self.deduplicator = SemanticDeduplicator(self.config)
        self.retriever = QueryAwareRetriever(self.store, self.config)
        self.prompt_builder = PromptBuilder()
        self.encoder = SPLEncoder()
        self.ollama = OllamaClient(base_url=self.config.ollama_base_url, model=self.config.ollama_model)

    def ingest_turns(self, turns: Iterable[str], session_id: str) -> None:
        existing = self.store.fetch_by_session(session_id)
        for turn in turns:
            new_smos = self.extractor.extract(turn, session_id=session_id)
            to_add, to_remove = self.deduplicator.deduplicate(new_smos, existing)
            if to_remove:
                removed_ids = set(to_remove)
                self.store.delete_many(to_remove)
                existing = [item for item in existing if item.id not in removed_ids]
            self.store.upsert_many(to_add)
            existing = to_add + existing

    def build_spl(self, query: str, session_id: str, top_k: int = 10) -> str:
        relevant = self.retriever.retrieve(query=query, session_id=session_id, top_k=top_k)
        return self.encoder.encode(relevant, query)

    def answer(self, query: str, session_id: str, top_k: int = 10) -> str:
        relevant = self.retriever.retrieve(query=query, session_id=session_id, top_k=top_k)
        prompt = self.prompt_builder.build(relevant, query)
        result = self.ollama.generate(prompt, system=self.prompt_builder.system_prompt)
        return result.get("response", "")
