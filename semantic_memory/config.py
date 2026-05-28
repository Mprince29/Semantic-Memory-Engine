from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
SQLITE_PATH = DATA_DIR / "semantic_memory.db"


@dataclass(slots=True)
class EngineConfig:
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_collection: str = "semantic_memory"
    allow_inmemory_vector_store: bool = False
    token_budget: int = 80
    similarity_threshold: float = 0.88
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    action_verbs: tuple[str, ...] = (
        "build",
        "create",
        "submit",
        "deploy",
        "fix",
        "write",
        "debug",
        "train",
        "containerize",
        "optimize",
        "implement",
        "migrate",
        "refactor",
        "test",
        "evaluate",
        "publish",
        "install",
        "configure",
        "design",
        "set up",
        "improve",
        "generate",
        "learn",
        "study",
        "prepare",
        "plan",
        "invest",
        "save",
        "reduce",
        "increase",
        "host",
        "run",
        "process",
        "integrate",
        "fine-tune",
        "compress",
        "classify",
        "transcribe",
        "extract",
        "deduplicate",
    )
    preference_indicators: dict[str, str] = field(
        default_factory=lambda: {
            "prefer": "pos",
            "like": "pos",
            "want": "pos",
            "need": "pos",
            "use": "pos",
            "hate": "neg",
            "avoid": "neg",
            "dislike": "neg",
            "without": "neg",
            "no": "neg",
            "instead of": "neg",
            "over": "pos",
            "focus on": "pos",
            "choose": "pos",
            "stick with": "pos",
            "keep": "pos",
            "skip": "neg",
        }
    )
    temporal_map: dict[str, str] = field(
        default_factory=lambda: {
            "tomorrow": "T+1",
            "today": "T+0",
            "yesterday": "T-1",
            "next week": "T+7",
            "this week": "T+0..T+7",
            "next month": "T+30",
            "this month": "T+0..T+30",
            "end of month": "T+30",
            "this friday": "T+5",
            "this thursday": "T+4",
            "this weekend": "T+5..T+7",
            "in 3 days": "T+3",
            "in 5 days": "T+5",
            "in 2 weeks": "T+14",
            "in 3 weeks": "T+21",
            "in 4 weeks": "T+28",
            "in 6 weeks": "T+42",
            "in 45 days": "T+45",
            "3 months": "T+90",
            "6 months": "T+180",
            "8 months": "T+240",
            "end of sprint": "T+14",
            "end of next sprint": "T+28",
        }
    )
    fine_tune_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    fine_tune_max_seq_length: int = 768
    fine_tune_output_dir: str = "artifacts/qwen2.5-3b-spl"

    # V2 — adaptive token budget tier overrides (0 = use analyzer default)
    budget_simple: int = 45
    budget_preference: int = 75
    budget_planning: int = 140

    # V2 — default SPL schema for new sessions
    default_schema: str = "general"

    # V2 — contradiction detection similarity floor (values below this are not comparable)
    contradiction_min_overlap: float = 0.1


DEFAULT_CONFIG = EngineConfig()
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
