import json
import sqlite3
from pathlib import Path

from semantic_memory.config import DEFAULT_CONFIG, EngineConfig, SQLITE_PATH, VECTOR_STORE_DIR
from semantic_memory.domain.models import (
    MEMORY_STATE_ACTIVE,
    VISIBILITY_PRIVATE,
    SemanticMemoryObject,
)

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None


class _InMemoryCollection:
    def __init__(self):
        self._records: dict[str, dict] = {}

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self._records[item_id] = {
                "embedding": [float(value) for value in embedding],
                "document": document,
                "metadata": dict(metadata),
            }

    def delete(self, ids: list[str]) -> None:
        for item_id in ids:
            self._records.pop(item_id, None)

    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str], where: dict | None = None) -> dict:
        query_embedding = [float(value) for value in query_embeddings[0]]
        candidates: list[tuple[float, dict]] = []
        for record in self._records.values():
            metadata = record["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            distance = 1.0 - self._cosine_similarity(query_embedding, record["embedding"])
            candidates.append((distance, metadata))

        candidates.sort(key=lambda item: item[0])
        top = candidates[:n_results]
        return {
            "metadatas": [[metadata for _, metadata in top]],
            "distances": [[distance for distance, _ in top]],
        }

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        denom = (left_norm * right_norm) + 1e-9
        dot = sum(a * b for a, b in zip(left, right))
        return float(dot / denom)


class SemanticMemoryStore:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.sqlite_path = Path(SQLITE_PATH)
        self.sqlite_path.parent.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(self.sqlite_path)
        self.connection.row_factory = sqlite3.Row
        self._init_db()
        if chromadb is None:
            if not self.config.allow_inmemory_vector_store:
                raise RuntimeError(
                    "ChromaDB is required for runtime use. "
                    "Install 'chromadb' or set allow_inmemory_vector_store=True for tests/dev only."
                )
            self.client = None
            self.collection = _InMemoryCollection()
        else:
            self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            self.collection = self.client.get_or_create_collection(self.config.vector_collection)

    def _init_db(self) -> None:
        self.connection.executescript(
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
                embedding TEXT NOT NULL,
                memory_state TEXT NOT NULL DEFAULT 'active',
                user_id TEXT NOT NULL DEFAULT '',
                visibility TEXT NOT NULL DEFAULT 'private',
                owner TEXT NOT NULL DEFAULT '',
                provenance TEXT NOT NULL DEFAULT ''
            );

            -- Federation table: shared memories across sessions under a user_id
            CREATE TABLE IF NOT EXISTS federated_memory (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                domain TEXT,
                deadline TEXT,
                confidence REAL NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                embedding TEXT NOT NULL,
                memory_state TEXT NOT NULL DEFAULT 'active',
                visibility TEXT NOT NULL DEFAULT 'team',
                owner TEXT NOT NULL DEFAULT '',
                provenance TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_session ON semantic_memory(session_id);
            CREATE INDEX IF NOT EXISTS idx_user ON semantic_memory(user_id);
            CREATE INDEX IF NOT EXISTS idx_fed_user ON federated_memory(user_id);
            """
        )
        self._migrate_add_v2_columns()
        self.connection.commit()

    def _migrate_add_v2_columns(self) -> None:
        """Idempotent migration: add V2 columns to existing databases."""
        existing = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(semantic_memory)").fetchall()
        }
        new_columns = {
            "memory_state": "TEXT NOT NULL DEFAULT 'active'",
            "user_id": "TEXT NOT NULL DEFAULT ''",
            "visibility": "TEXT NOT NULL DEFAULT 'private'",
            "owner": "TEXT NOT NULL DEFAULT ''",
            "provenance": "TEXT NOT NULL DEFAULT ''",
        }
        for col, definition in new_columns.items():
            if col not in existing:
                self.connection.execute(
                    f"ALTER TABLE semantic_memory ADD COLUMN {col} {definition}"
                )

    # ------------------------------------------------------------------ write

    def upsert_many(self, smos: list[SemanticMemoryObject]) -> None:
        if not smos:
            return
        smos = list({smo.id: smo for smo in smos}.values())

        self.connection.executemany(
            """
            INSERT OR REPLACE INTO semantic_memory
            (id, type, subject, predicate, value, domain, deadline, confidence,
             session_id, timestamp, embedding, memory_state, user_id, visibility, owner, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    smo.id, smo.type, smo.subject, smo.predicate, smo.value,
                    smo.domain, smo.deadline, smo.confidence, smo.session_id,
                    smo.timestamp, json.dumps(smo.embedding),
                    smo.memory_state, smo.user_id, smo.visibility,
                    smo.owner, smo.provenance,
                )
                for smo in smos
            ],
        )
        self.connection.commit()

        self._upsert_collection_records(smos)

    def update_memory_state(self, smo_id: str, state: str) -> None:
        # Keep both tables in sync — a disputed/superseded memory must not
        # appear as active in the federation pool either
        self.connection.execute(
            "UPDATE semantic_memory SET memory_state = ? WHERE id = ?",
            (state, smo_id),
        )
        self.connection.execute(
            "UPDATE federated_memory SET memory_state = ? WHERE id = ?",
            (state, smo_id),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM semantic_memory WHERE id = ?",
            (smo_id,),
        ).fetchone()
        if row is None:
            return
        smo = self._row_to_smo(row)
        self._upsert_collection_records([smo])

    def delete_many(self, ids: list[str]) -> None:
        if not ids:
            return
        unique_ids = list(dict.fromkeys(ids))
        self.connection.executemany(
            "DELETE FROM semantic_memory WHERE id = ?",
            [(item_id,) for item_id in unique_ids],
        )
        self.connection.commit()
        self.collection.delete(ids=unique_ids)

    # ------------------------------------------------------------------ read

    def fetch_by_session(self, session_id: str) -> list[SemanticMemoryObject]:
        rows = self.connection.execute(
            "SELECT * FROM semantic_memory WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,),
        ).fetchall()
        return [self._row_to_smo(row) for row in rows]

    def clear_session(self, session_id: str) -> int:
        """Delete all memories for a session. Returns the number of records removed."""
        ids = [row["id"] for row in self.connection.execute(
            "SELECT id FROM semantic_memory WHERE session_id = ?", (session_id,)
        ).fetchall()]
        if ids:
            self.connection.execute(
                "DELETE FROM semantic_memory WHERE session_id = ?", (session_id,)
            )
            self.connection.execute(
                "DELETE FROM federated_memory WHERE session_id = ?", (session_id,)
            )
            self.connection.commit()
            try:
                self.collection.delete(ids=ids)
            except Exception:
                pass
        return len(ids)

    def fetch_contextual_by_session(self, session_id: str) -> list[SemanticMemoryObject]:
        rows = self.connection.execute(
            "SELECT * FROM semantic_memory WHERE session_id = ? AND memory_state = 'contextual' ORDER BY timestamp DESC",
            (session_id,),
        ).fetchall()
        return [self._row_to_smo(row) for row in rows]

    def fetch_active_by_session(self, session_id: str) -> list[SemanticMemoryObject]:
        rows = self.connection.execute(
            "SELECT * FROM semantic_memory WHERE session_id = ? AND memory_state = ? ORDER BY timestamp DESC",
            (session_id, MEMORY_STATE_ACTIVE),
        ).fetchall()
        return [self._row_to_smo(row) for row in rows]

    # ------------------------------------------------------------------ federation

    def publish_to_federation(self, smos: list[SemanticMemoryObject]) -> None:
        """Publish memories with team/public visibility to the shared federation pool."""
        shareable = [s for s in smos if s.visibility in ("team", "public") and s.user_id]
        if not shareable:
            return
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO federated_memory
            (id, type, subject, predicate, value, domain, deadline, confidence,
             user_id, session_id, timestamp, embedding, memory_state, visibility, owner, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.id, s.type, s.subject, s.predicate, s.value,
                    s.domain, s.deadline, s.confidence, s.user_id, s.session_id,
                    s.timestamp, json.dumps(s.embedding),
                    s.memory_state, s.visibility, s.owner, s.provenance,
                )
                for s in shareable
            ],
        )
        self.connection.commit()

    def fetch_federated(
        self, user_id: str, exclude_session: str = ""
    ) -> list[SemanticMemoryObject]:
        """Fetch shared memories for a user from sessions other than the current one."""
        rows = self.connection.execute(
            """
            SELECT * FROM federated_memory
            WHERE user_id = ? AND session_id != ? AND memory_state = 'active'
            ORDER BY timestamp DESC
            """,
            (user_id, exclude_session),
        ).fetchall()
        return [self._row_to_smo_federated(row) for row in rows]

    # ------------------------------------------------------------------ helpers

    def _metadata_for_chroma(self, smo: SemanticMemoryObject) -> dict:
        metadata = smo.to_metadata()
        metadata["embedding"] = json.dumps(metadata["embedding"])
        # ChromaDB rejects None values — replace with empty string sentinel
        for key, val in metadata.items():
            if val is None:
                metadata[key] = ""
        return metadata

    def rebuild_vector_index(self, embedding_dim: int) -> None:
        """Rebuild the Chroma collection using only rows that match the target dimension."""
        if self.client is None:
            self.collection = _InMemoryCollection()
            for row in self.connection.execute(
                "SELECT * FROM semantic_memory ORDER BY timestamp DESC"
            ).fetchall():
                smo = self._row_to_smo(row)
                if len(smo.embedding) != embedding_dim:
                    continue
                self.collection.upsert(
                    ids=[smo.id],
                    embeddings=[smo.embedding],
                    documents=[smo.text_for_embedding()],
                    metadatas=[self._metadata_for_chroma(smo)],
                )
            return

        try:
            self.client.delete_collection(name=self.config.vector_collection)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.config.vector_collection)

        rows = self.connection.execute(
            "SELECT * FROM semantic_memory ORDER BY timestamp DESC"
        ).fetchall()
        smos = [
            self._row_to_smo(row)
            for row in rows
            if len(json.loads(row["embedding"])) == embedding_dim
        ]
        if smos:
            self.collection.upsert(
                ids=[smo.id for smo in smos],
                embeddings=[smo.embedding for smo in smos],
                documents=[smo.text_for_embedding() for smo in smos],
                metadatas=[self._metadata_for_chroma(smo) for smo in smos],
            )

    def _upsert_collection_records(self, smos: list[SemanticMemoryObject]) -> None:
        smos = list({smo.id: smo for smo in smos}.values())
        try:
            self.collection.upsert(
                ids=[smo.id for smo in smos],
                embeddings=[smo.embedding for smo in smos],
                documents=[smo.text_for_embedding() for smo in smos],
                metadatas=[self._metadata_for_chroma(smo) for smo in smos],
            )
        except Exception as exc:
            # Persistent stores may contain older embeddings with a different size.
            # Rebuild the vector index for the current embedding dimension, then retry.
            if not self._is_dimension_mismatch(exc):
                raise
            self.rebuild_vector_index(len(smos[0].embedding))
            self.collection.upsert(
                ids=[smo.id for smo in smos],
                embeddings=[smo.embedding for smo in smos],
                documents=[smo.text_for_embedding() for smo in smos],
                metadatas=[self._metadata_for_chroma(smo) for smo in smos],
            )

    @staticmethod
    def _is_dimension_mismatch(exc: Exception) -> bool:
        message = str(exc).lower()
        return "dimension" in message and ("expecting embedding" in message or "not aligned" in message)

    @staticmethod
    def _row_to_smo(row: sqlite3.Row) -> SemanticMemoryObject:
        d = dict(row)
        return SemanticMemoryObject(
            id=d["id"],
            type=d["type"],
            subject=d["subject"],
            predicate=d["predicate"],
            value=d["value"],
            domain=d.get("domain"),
            deadline=d.get("deadline"),
            confidence=d["confidence"],
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            embedding=json.loads(d["embedding"]),
            memory_state=d.get("memory_state", MEMORY_STATE_ACTIVE),
            user_id=d.get("user_id", ""),
            visibility=d.get("visibility", VISIBILITY_PRIVATE),
            owner=d.get("owner", ""),
            provenance=d.get("provenance", ""),
        )

    @staticmethod
    def _row_to_smo_federated(row: sqlite3.Row) -> SemanticMemoryObject:
        d = dict(row)
        return SemanticMemoryObject(
            id=d["id"],
            type=d["type"],
            subject=d["subject"],
            predicate=d["predicate"],
            value=d["value"],
            domain=d.get("domain"),
            deadline=d.get("deadline"),
            confidence=d["confidence"],
            session_id=d.get("session_id", ""),
            timestamp=d["timestamp"],
            embedding=json.loads(d["embedding"]),
            memory_state=d.get("memory_state", MEMORY_STATE_ACTIVE),
            user_id=d.get("user_id", ""),
            visibility=d.get("visibility", "team"),
            owner=d.get("owner", ""),
            provenance=d.get("provenance", ""),
        )
