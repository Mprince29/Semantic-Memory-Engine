from dataclasses import asdict, dataclass, field
from typing import Any


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

    def text_for_embedding(self) -> str:
        return " ".join(part for part in (self.subject, self.predicate, self.value) if part)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["embedding"] = list(self.embedding)
        return data
