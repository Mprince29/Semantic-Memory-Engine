import json
import sqlite3
from pathlib import Path

import chromadb

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig, SQLITE_PATH, VECTOR_STORE_DIR
from semantic_memory.domain.models import SemanticMemoryObject


class SemanticMemoryStore:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.sqlite_path = Path(SQLITE_PATH)
        self.sqlite_path.parent.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(self.sqlite_path)
        self.connection.row_factory = sqlite3.Row
        self._init_db()
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.client.get_or_create_collection(self.config.vector_collection)

    def _init_db(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                domain TEXT,
                deadline TEXT,
                confidence REAL NOT NULL,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def upsert_many(self, smos: list[SemanticMemoryObject]) -> None:
        if not smos:
            return

        self.connection.executemany(
            """
            INSERT OR REPLACE INTO semantic_memory
            (id, type, subject, predicate, value, domain, deadline, confidence, session_id, timestamp, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    smo.id,
                    smo.type,
                    smo.subject,
                    smo.predicate,
                    smo.value,
                    smo.domain,
                    smo.deadline,
                    smo.confidence,
                    smo.session_id,
                    smo.timestamp,
                    json.dumps(smo.embedding),
                )
                for smo in smos
            ],
        )
        self.connection.commit()

        self.collection.upsert(
            ids=[smo.id for smo in smos],
            embeddings=[smo.embedding for smo in smos],
            documents=[smo.text_for_embedding() for smo in smos],
            metadatas=[self._metadata_for_chroma(smo) for smo in smos],
        )

    def delete_many(self, ids: list[str]) -> None:
        if not ids:
            return
        self.connection.executemany("DELETE FROM semantic_memory WHERE id = ?", [(item_id,) for item_id in ids])
        self.connection.commit()
        self.collection.delete(ids=ids)

    def fetch_by_session(self, session_id: str) -> list[SemanticMemoryObject]:
        rows = self.connection.execute(
            "SELECT * FROM semantic_memory WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,),
        ).fetchall()
        return [self._row_to_smo(row) for row in rows]

    def _metadata_for_chroma(self, smo: SemanticMemoryObject) -> dict:
        metadata = smo.to_metadata()
        metadata["embedding"] = json.dumps(metadata["embedding"])
        return metadata

    @staticmethod
    def _row_to_smo(row: sqlite3.Row) -> SemanticMemoryObject:
        return SemanticMemoryObject(
            id=row["id"],
            type=row["type"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            domain=row["domain"],
            deadline=row["deadline"],
            confidence=row["confidence"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            embedding=json.loads(row["embedding"]),
        )
