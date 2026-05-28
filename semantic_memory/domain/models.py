from dataclasses import asdict, dataclass, field
from typing import Any

# Memory lifecycle states for contradiction-aware compression
MEMORY_STATE_ACTIVE = "active"
MEMORY_STATE_SUPERSEDED = "superseded"   # overwritten by a newer, compatible memory
MEMORY_STATE_CONTEXTUAL = "contextual"   # scoped truth (e.g. "at work" vs "for this project")
MEMORY_STATE_DISPUTED = "disputed"       # direct polarity conflict, not yet resolved

# Visibility scopes for cross-session federation
VISIBILITY_PRIVATE = "private"
VISIBILITY_TEAM = "team"
VISIBILITY_PUBLIC = "public"


@dataclass(slots=True)
class SemanticMemoryObject:
    id: str
    type: str
    subject: str
    predicate: str
    value: str
    domain: str | None = None
    deadline: str | None = None
    confidence: float = 1.0
    session_id: str = ""
    timestamp: float = 0.0
    embedding: list[float] = field(default_factory=list)

    # V2 fields
    memory_state: str = MEMORY_STATE_ACTIVE
    user_id: str = ""
    visibility: str = VISIBILITY_PRIVATE
    owner: str = ""
    provenance: str = ""           # source session or agent that created this memory

    def text_for_embedding(self) -> str:
        return " ".join(part for part in (self.subject, self.predicate, self.value) if part)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["embedding"] = list(self.embedding)
        return data

    def is_active(self) -> bool:
        return self.memory_state == MEMORY_STATE_ACTIVE
