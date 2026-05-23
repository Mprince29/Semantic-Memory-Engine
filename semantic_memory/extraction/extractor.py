import hashlib
import re
import time

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import SemanticMemoryObject

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class HashEmbedder:
    def __init__(self, size: int = 64):
        self.size = size

    def encode(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.size:
            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)
                if len(values) >= self.size:
                    break
        return values


class SemanticExtractor:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.nlp = self._load_nlp(self.config.spacy_model)
        self.embedder = self._load_embedder(self.config.embedding_model)

    def _load_nlp(self, model_name: str):
        if spacy is None:
            return None
        try:
            return spacy.load(model_name)
        except OSError:
            return spacy.blank("en")

    def _load_embedder(self, model_name: str):
        if SentenceTransformer is None:
            return HashEmbedder()
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return HashEmbedder()

    def extract(self, text: str, session_id: str) -> list[SemanticMemoryObject]:
        now = time.time()
        smos: list[SemanticMemoryObject] = []
        doc = self.nlp(text) if self.nlp else None

        if doc and hasattr(doc, "ents"):
            for ent in doc.ents:
                smos.append(
                    SemanticMemoryObject(
                        id=self._make_id("entity", ent.text, session_id),
                        type="entity",
                        subject=ent.text,
                        predicate="is_a",
                        value=ent.label_,
                        session_id=session_id,
                        timestamp=now,
                    )
                )

        smos.extend(self._extract_tasks(text, session_id, now))
        smos.extend(self._extract_preferences(text, session_id, now))
        smos.extend(self._extract_temporal(text, session_id, now))
        smos.extend(self._extract_facts(text, session_id, now))
        smos.extend(self._extract_topics(text, session_id, now))

        for smo in smos:
            vector = self.embedder.encode(smo.text_for_embedding())
            smo.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)

        return smos

    def _extract_tasks(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for verb in self.config.action_verbs:
            pattern = rf"\b{re.escape(verb)}\b\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+){{0,3}})"
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = match.group(1).strip(" .,!?")
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("task", f"{verb}:{value}", session_id),
                    type="task",
                    subject="user",
                    predicate=verb,
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )
        return smos

    def _extract_preferences(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for keyword, polarity in self.config.preference_indicators.items():
            pattern = rf"\b{re.escape(keyword)}\b\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+){{0,3}})"
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = match.group(1).strip(" .,!?")
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("preference", f"{polarity}:{value}", session_id),
                    type="preference",
                    subject="user",
                    predicate=f"pref_{polarity}",
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )
        return smos

    def _extract_temporal(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for phrase, tcode in self.config.temporal_map.items():
            if phrase not in lowered:
                continue
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("constraint", f"deadline:{tcode}", session_id),
                    type="constraint",
                    subject="task",
                    predicate="deadline",
                    value=tcode,
                    deadline=tcode,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )
        return smos

    def _extract_topics(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        keywords = (
            # infra / ML
            "ollama", "docker", "rag", "embedding", "chromadb", "python", "debug", "spl",
            "fastapi", "django", "flask", "postgres", "postgresql", "redis", "kafka",
            "kubernetes", "nginx", "gunicorn", "celery", "sqlite", "react", "typescript",
            "pytorch", "tensorflow", "scikit-learn", "pandas", "polars", "numpy", "spacy",
            "transformers", "huggingface", "lora", "qlora", "llm", "finetune", "fine-tune",
            "quantization", "inference", "chromadb", "faiss", "qdrant", "weaviate",
            "whisper", "bert", "qwen", "llama", "mistral", "colab", "gpu", "cpu",
            "github", "git", "ci", "cd", "pytest", "unittest", "benchmark",
            # domains
            "exam", "gate", "interview", "thesis", "research", "dataset",
            "investment", "budget", "savings", "startup", "mvp", "hiring",
            "workout", "diet", "health", "cooking", "travel", "trip",
            "android", "mobile", "api", "microservice", "monolith",
            "compression", "token", "context", "memory", "retrieval", "semantic",
            "classification", "nlp", "graph", "knowledge", "sql", "database",
        )
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                smos.append(
                    SemanticMemoryObject(
                        id=self._make_id("event", f"topic:{keyword}", session_id),
                        type="event",
                        subject="conversation",
                        predicate="topic",
                        value=keyword,
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                )
        return smos

    def _extract_facts(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()

        ram_match = re.search(r"\b(\d+)\s*gb\s*ram\b", lowered)
        if ram_match:
            value = f"{ram_match.group(1)}gb_ram"
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", f"hw:{value}", session_id),
                    type="fact",
                    subject="user",
                    predicate="hw",
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        status_patterns = {
            "debugging": "debug",
            "in progress": "in_progress",
            "working on": "in_progress",
            "stuck": "blocked",
            "currently profiling": "profiling",
            "currently load testing": "load_testing",
            "currently evaluating": "evaluating",
            "evaluation phase": "evaluating",
            "design phase": "designing",
            "currently debugging": "debug",
            "currently in": "in_progress",
        }
        for phrase, status in status_patterns.items():
            if phrase not in lowered:
                continue
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", f"status:{status}", session_id),
                    type="fact",
                    subject="task",
                    predicate="status",
                    value=status,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        scope_patterns = {
            "local only": "local_only",
            "local-only": "local_only",
            "self-hosted": "self_hosted",
            "self hosted": "self_hosted",
            "on-premise": "on_premise",
            "offline": "offline_only",
            "no cloud": "local_only",
            "avoid cloud": "local_only",
        }
        for phrase, scope_val in scope_patterns.items():
            if phrase in lowered:
                smos.append(
                    SemanticMemoryObject(
                        id=self._make_id("fact", f"scope:{scope_val}", session_id),
                        type="fact",
                        subject="task",
                        predicate="scope",
                        value=scope_val,
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                )
                break

        # Budget extraction (Indian rupees)
        budget_match = re.search(r"\b(\d+)\s*(?:lpa|rupees?|rs\.?|₹)\b", lowered)
        if budget_match:
            value = f"{budget_match.group(1)}rupees"
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", f"budget:{value}", session_id),
                    type="fact",
                    subject="user",
                    predicate="budget",
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        # vCPU / core count
        vcpu_match = re.search(r"\b(\d+)\s*v?cpu[s]?\b", lowered)
        if vcpu_match:
            value = f"{vcpu_match.group(1)}vcpu"
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", f"hw:{value}", session_id),
                    type="fact",
                    subject="user",
                    predicate="hw",
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        # GPU availability
        if re.search(r"\bno\s+gpu\b", lowered):
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", "hw:no_gpu", session_id),
                    type="fact",
                    subject="user",
                    predicate="hw",
                    value="no_gpu",
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )
        elif re.search(r"\b(?:t4|a100|v100|gpu)\b", lowered):
            gpu_match = re.search(r"\b(t4|a100|v100)\b", lowered)
            gpu_val = gpu_match.group(1) if gpu_match else "gpu"
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("fact", f"hw:{gpu_val}", session_id),
                    type="fact",
                    subject="user",
                    predicate="hw",
                    value=gpu_val,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        return smos

    @staticmethod
    def _make_id(prefix: str, value: str, session_id: str) -> str:
        digest = hashlib.md5(f"{prefix}:{value}:{session_id}".encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"
