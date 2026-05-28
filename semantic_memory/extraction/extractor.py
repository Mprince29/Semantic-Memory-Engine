import hashlib
import re
import time

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig
from semantic_memory.domain.models import SemanticMemoryObject


_CODING_STACK_KEYWORDS = (
    "flask", "fastapi", "django", "gunicorn", "uvicorn", "nginx", "docker",
    "postgres", "postgresql", "sqlite", "redis", "celery", "react",
    "typescript", "python", "kubernetes", "chromadb", "ollama",
)

_TASK_LEADING_STOPWORDS = frozenset({
    "a", "an", "the", "to", "my", "our", "your", "this", "that",
})

_TASK_TRAILING_STOPWORDS = frozenset({
    "today", "tomorrow", "tonight", "now", "soon", "later",
})

_PREFERENCE_VALUE_HINTS = frozenset(_CODING_STACK_KEYWORDS) | {
    "cloud", "cloud services", "local", "offline", "systemd", "api",
    "postgres", "redis", "gpu", "cpu",
}

_DEBUG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b500\b.*\bstartup\b|\bstartup\b.*\b500\b"), "500_on_startup"),
    (re.compile(r"\b404\b"), "404_error"),
    (re.compile(r"\b500\b"), "500_error"),
    (re.compile(r"\btimeout\b|\btimed out\b"), "timeout"),
    (re.compile(r"\btraceback\b"), "traceback"),
    (re.compile(r"\bstartup failure\b|\bfails? on startup\b|\bcrash(?:es)? on startup\b"), "startup_failure"),
    (re.compile(r"\bconnection refused\b"), "connection_refused"),
    (re.compile(r"\bmodule not found\b|\bmodulenotfounderror\b"), "module_not_found"),
    (re.compile(r"\bimport error\b|\bimporterror\b"), "import_error"),
    (re.compile(r"\bsegmentation fault\b|\bsegfault\b"), "segfault"),
    (re.compile(r"\bexception\b"), "exception"),
    (re.compile(r"\bcrash(?:ed|es|ing)?\b"), "crash"),
    (re.compile(r"\bdebug(?:ging)?\b"), "debugging"),
)

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
        smos.extend(self._extract_coding_stack(text, session_id, now))
        smos.extend(self._extract_debug_signals(text, session_id, now))
        smos.extend(self._extract_topics(text, session_id, now))

        for smo in smos:
            vector = self.embedder.encode(smo.text_for_embedding())
            smo.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)

        return smos

    def _extract_tasks(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for verb in self.config.action_verbs:
            pattern = rf"\b{re.escape(verb)}\b\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+){{0,5}})"
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = self._clean_task_value(match.group(1))
            if not value:
                continue
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

    # Preference indicators that fire too broadly on non-preference sentences.
    # These are valid signals only when followed by a tool/technology/concept,
    # not by verbs, articles, or continuation phrases.
    _WEAK_PREF_INDICATORS = frozenset({"use", "need", "want", "keep", "no", "without"})
    # First token of a match must not be one of these — catches "getting", "to", "it", etc.
    _PREF_VALUE_BLOCKLIST = frozenset({
        "a", "an", "the", "to", "it", "this", "that", "my", "your", "our",
        "getting", "going", "being", "doing", "having", "running",
        "up", "out", "on", "in", "at", "by", "as", "so",
    })

    def _extract_preferences(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        for keyword, polarity in self.config.preference_indicators.items():
            pattern = rf"\b{re.escape(keyword)}\b\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+){{0,3}})"
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = match.group(1).strip(" .,!?")
            first_token = value.split()[0] if value.split() else ""

            # Weak indicators need the extracted value to look like a real noun/tool,
            # not a continuation of a sentence ("keep it local only", "need to deploy")
            if keyword in self._WEAK_PREF_INDICATORS and first_token in self._PREF_VALUE_BLOCKLIST:
                continue

            # Value must be at least one non-trivial token
            if first_token in self._PREF_VALUE_BLOCKLIST:
                continue

            if keyword in self._WEAK_PREF_INDICATORS and not self._looks_like_preference_value(value):
                continue

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

    def _extract_coding_stack(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        seen: set[str] = set()

        for keyword in _CODING_STACK_KEYWORDS:
            if keyword not in lowered:
                continue
            value = "postgres" if keyword == "postgresql" else keyword
            if value in seen:
                continue
            seen.add(value)
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("stack", value, session_id),
                    type="stack",
                    subject="app",
                    predicate="uses",
                    value=value,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        return smos

    def _extract_debug_signals(self, text: str, session_id: str, timestamp: float) -> list[SemanticMemoryObject]:
        smos: list[SemanticMemoryObject] = []
        lowered = text.lower()
        seen: set[str] = set()

        for pattern, label in _DEBUG_PATTERNS:
            if not pattern.search(lowered):
                continue
            if label in seen:
                continue
            seen.add(label)
            smos.append(
                SemanticMemoryObject(
                    id=self._make_id("debug", label, session_id),
                    type="debug",
                    subject="app",
                    predicate="error",
                    value=label,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )

        if "500_on_startup" in seen:
            smos = [item for item in smos if item.value != "500_error"]
        if "startup_failure" in seen and "500_on_startup" in seen:
            smos = [item for item in smos if item.value != "startup_failure"]

        return smos

    @staticmethod
    def _clean_task_value(raw_value: str) -> str:
        value = raw_value.strip(" .,!?")
        words = value.split()
        while words and words[0] in _TASK_LEADING_STOPWORDS:
            words.pop(0)
        while words and words[-1] in _TASK_TRAILING_STOPWORDS:
            words.pop()
        if len(words) >= 2 and words[-2:] == ["next", "week"]:
            words = words[:-2]
        if len(words) >= 2 and words[-2:] == ["this", "week"]:
            words = words[:-2]
        if len(words) >= 3 and words[-3:] == ["for", "this", "project"]:
            words = words[:-3]
        return " ".join(words).strip()

    @staticmethod
    def _looks_like_preference_value(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in _PREFERENCE_VALUE_HINTS:
            return True
        tokens = [token for token in normalized.replace("_", " ").split() if token]
        if not tokens:
            return False
        return any(token in _PREFERENCE_VALUE_HINTS for token in tokens)

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
