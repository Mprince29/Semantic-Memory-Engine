import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_memory.application.engine import SemanticRuntimeEngine
from semantic_memory.config import EngineConfig
from semantic_memory.contradiction.detector import ContradictionDetector
from semantic_memory.domain.models import (
    MEMORY_STATE_ACTIVE,
    MEMORY_STATE_CONTEXTUAL,
    MEMORY_STATE_DISPUTED,
    MEMORY_STATE_SUPERSEDED,
    VISIBILITY_TEAM,
    SemanticMemoryObject,
)
from semantic_memory.extraction.extractor import HashEmbedder, SemanticExtractor
from semantic_memory.infrastructure import store as store_module
from semantic_memory.infrastructure.store import SemanticMemoryStore
from semantic_memory.retrieval.complexity import QueryComplexityAnalyzer


def make_smo(
    *,
    item_id: str,
    item_type: str,
    predicate: str,
    value: str,
    subject: str = "user",
    session_id: str = "s1",
    timestamp: float = 1.0,
    user_id: str = "",
    visibility: str = "private",
    memory_state: str = MEMORY_STATE_ACTIVE,
) -> SemanticMemoryObject:
    embedder = HashEmbedder()
    text = " ".join(part for part in (subject, predicate, value) if part)
    return SemanticMemoryObject(
        id=item_id,
        type=item_type,
        subject=subject,
        predicate=predicate,
        value=value,
        session_id=session_id,
        timestamp=timestamp,
        user_id=user_id,
        visibility=visibility,
        memory_state=memory_state,
        embedding=embedder.encode(text),
    )


class StoreIsolationTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.sqlite_path = Path(self.temp_dir.name) / "semantic_memory.db"
        self.vector_store_dir = Path(self.temp_dir.name) / "vector_store"
        self.vector_store_dir.mkdir(exist_ok=True)

        self.sqlite_patch = patch.object(store_module, "SQLITE_PATH", self.sqlite_path)
        self.vector_patch = patch.object(store_module, "VECTOR_STORE_DIR", self.vector_store_dir)
        self.sqlite_patch.start()
        self.vector_patch.start()
        self.addCleanup(self.sqlite_patch.stop)
        self.addCleanup(self.vector_patch.stop)

    @staticmethod
    def make_test_config(**overrides) -> EngineConfig:
        return EngineConfig(
            allow_inmemory_vector_store=(store_module.chromadb is None),
            **overrides,
        )


class ContradictionDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = ContradictionDetector()

    def test_marks_scoped_preference_flip_as_contextual(self):
        existing = make_smo(
            item_id="old",
            item_type="preference",
            predicate="pref_neg",
            value="docker",
            timestamp=1.0,
        )
        candidate = make_smo(
            item_id="new",
            item_type="preference",
            predicate="pref_pos",
            value="docker",
            timestamp=2.0,
        )

        result = self.detector.check(candidate, [existing], source_text="At work, I prefer Docker.")

        self.assertEqual(result.resolution, "contextual")
        self.assertEqual(result.state_for_candidate, MEMORY_STATE_CONTEXTUAL)
        self.assertEqual(result.state_for_conflicting, MEMORY_STATE_CONTEXTUAL)

    def test_marks_unscoped_preference_flip_as_disputed(self):
        existing = make_smo(
            item_id="old",
            item_type="preference",
            predicate="pref_neg",
            value="docker",
            timestamp=1.0,
        )
        candidate = make_smo(
            item_id="new",
            item_type="preference",
            predicate="pref_pos",
            value="docker",
            timestamp=2.0,
        )

        result = self.detector.check(candidate, [existing], source_text="I prefer Docker now.")

        self.assertEqual(result.resolution, "disputed")
        self.assertEqual(result.state_for_candidate, MEMORY_STATE_ACTIVE)
        self.assertEqual(result.state_for_conflicting, MEMORY_STATE_DISPUTED)

    def test_marks_fact_updates_as_superseded(self):
        existing = make_smo(
            item_id="old",
            item_type="fact",
            predicate="hw",
            value="16gb_ram",
            timestamp=1.0,
        )
        candidate = make_smo(
            item_id="new",
            item_type="fact",
            predicate="hw",
            value="32gb_ram",
            timestamp=2.0,
        )

        result = self.detector.check(candidate, [existing], source_text="I upgraded to 32GB RAM.")

        self.assertEqual(result.resolution, "superseded")
        self.assertEqual(result.state_for_candidate, MEMORY_STATE_ACTIVE)
        self.assertEqual(result.state_for_conflicting, MEMORY_STATE_SUPERSEDED)


class QueryComplexityAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = QueryComplexityAnalyzer()

    def test_simple_tier_for_short_lookup(self):
        result = self.analyzer.analyze("What is my deadline?")
        self.assertEqual(result.tier, "simple")
        self.assertEqual(result.token_budget, 45)

    def test_preference_tier_for_setup_question(self):
        result = self.analyzer.analyze("Which stack should I use for my project?")
        self.assertEqual(result.tier, "preference")
        self.assertEqual(result.token_budget, 75)

    def test_planning_tier_for_complex_debug_query(self):
        result = self.analyzer.analyze(
            "How should I debug and compare deployment options for my Flask app, "
            "and what architecture should I use to ship next week?"
        )
        self.assertEqual(result.tier, "planning")
        self.assertEqual(result.token_budget, 140)
        self.assertTrue(any(signal.startswith("planning:") for signal in result.signals))

    def test_budget_overrides_are_taken_from_config(self):
        analyzer = QueryComplexityAnalyzer(
            config=EngineConfig(
                budget_simple=11,
                budget_preference=22,
                budget_planning=33,
            )
        )

        self.assertEqual(analyzer.analyze("What is my deadline?").token_budget, 11)
        self.assertEqual(
            analyzer.analyze("Which stack should I use for my project?").token_budget,
            22,
        )
        self.assertEqual(
            analyzer.analyze(
                "How should I debug and compare deployment options for my Flask app next week?"
            ).token_budget,
            33,
        )


class ContradictionConfigTests(unittest.TestCase):
    def test_min_overlap_config_controls_preference_conflict_detection(self):
        candidate = make_smo(
            item_id="new",
            item_type="preference",
            predicate="pref_pos",
            value="docker compose",
            timestamp=2.0,
        )
        existing = make_smo(
            item_id="old",
            item_type="preference",
            predicate="pref_neg",
            value="docker swarm",
            timestamp=1.0,
        )

        permissive = ContradictionDetector(
            config=EngineConfig(contradiction_min_overlap=0.1)
        )
        strict = ContradictionDetector(
            config=EngineConfig(contradiction_min_overlap=0.6)
        )

        self.assertEqual(
            permissive.check(candidate, [existing], source_text="I prefer docker compose.").resolution,
            "disputed",
        )
        self.assertEqual(
            strict.check(candidate, [existing], source_text="I prefer docker compose.").resolution,
            "none",
        )


class SemanticExtractorV2Tests(unittest.TestCase):
    def setUp(self):
        self.extractor = SemanticExtractor()

    def test_extracts_coding_stack_and_debug_slots_from_plain_text(self):
        smos = self.extractor.extract(
            (
                "I am deploying a Flask API with Gunicorn and Nginx, "
                "and I keep getting a 500 on startup with a traceback."
            ),
            session_id="s1",
        )

        stacks = {item.value for item in smos if item.type == "stack"}
        debug = {item.value for item in smos if item.type == "debug"}

        self.assertTrue({"flask", "gunicorn", "nginx"}.issubset(stacks))
        self.assertIn("500_on_startup", debug)
        self.assertIn("traceback", debug)
        self.assertNotIn("500_error", debug)

    def test_task_extraction_is_cleaner_and_skips_fake_preferences(self):
        smos = self.extractor.extract(
            "I need to deploy a Flask API tomorrow.",
            session_id="s1",
        )

        tasks = [(item.predicate, item.value) for item in smos if item.type == "task"]
        prefs = [(item.predicate, item.value) for item in smos if item.type == "preference"]

        self.assertIn(("deploy", "flask api"), tasks)
        self.assertEqual(prefs, [])

    def test_keep_local_only_does_not_become_positive_preference(self):
        smos = self.extractor.extract(
            "Please keep it local only and avoid cloud services.",
            session_id="s1",
        )

        prefs = {(item.predicate, item.value) for item in smos if item.type == "preference"}

        self.assertNotIn(("pref_pos", "it local only"), prefs)
        self.assertIn(("pref_neg", "cloud services"), prefs)


class FederationStoreTests(StoreIsolationTestCase):
    def test_publish_fetch_and_state_sync_across_tables(self):
        store = SemanticMemoryStore(config=self.make_test_config())
        shareable = make_smo(
            item_id="shared-1",
            item_type="fact",
            predicate="scope",
            value="local_only",
            session_id="s1",
            timestamp=3.0,
            user_id="u1",
            visibility=VISIBILITY_TEAM,
        )

        store.upsert_many([shareable])
        store.publish_to_federation([shareable])

        fed = store.fetch_federated(user_id="u1", exclude_session="s2")
        self.assertEqual([item.id for item in fed], ["shared-1"])

        store.update_memory_state("shared-1", MEMORY_STATE_SUPERSEDED)
        fed_after = store.fetch_federated(user_id="u1", exclude_session="s2")
        self.assertEqual(fed_after, [])

        session_rows = store.fetch_by_session("s1")
        self.assertEqual(session_rows[0].memory_state, MEMORY_STATE_SUPERSEDED)

    def test_inmemory_fallback_requires_explicit_opt_in(self):
        with patch.object(store_module, "chromadb", None):
            with self.assertRaises(RuntimeError):
                SemanticMemoryStore(config=EngineConfig())

            store = SemanticMemoryStore(
                config=EngineConfig(allow_inmemory_vector_store=True)
            )
            self.assertIsNotNone(store.collection)


class RuntimeIntegrationTests(StoreIsolationTestCase):
    def test_end_to_end_v2_smoke_without_ollama(self):
        config = self.make_test_config(similarity_threshold=0.999)
        engine = SemanticRuntimeEngine(config=config)
        engine.extractor.embedder = HashEmbedder()
        engine.retriever.embedder = HashEmbedder()

        scripted_turns = {
            "I need to deploy a Flask API tomorrow, prefer nginx, avoid docker, and I am debugging a 500 startup failure on Flask plus Gunicorn with 16GB RAM.": [
                make_smo(item_id="task-1", item_type="task", predicate="deploy", value="flask api", session_id="s1", timestamp=1.0),
                make_smo(item_id="deadline-1", item_type="constraint", predicate="deadline", value="T+1", subject="task", session_id="s1", timestamp=1.0),
                make_smo(item_id="pref-pos-1", item_type="preference", predicate="pref_pos", value="nginx", session_id="s1", timestamp=1.0),
                make_smo(item_id="pref-neg-1", item_type="preference", predicate="pref_neg", value="docker", session_id="s1", timestamp=1.0),
                make_smo(item_id="fact-1", item_type="fact", predicate="hw", value="16gb_ram", session_id="s1", timestamp=1.0),
                make_smo(item_id="stack-1", item_type="stack", predicate="uses", value="flask", subject="app", session_id="s1", timestamp=1.0),
                make_smo(item_id="stack-2", item_type="stack", predicate="uses", value="gunicorn", subject="app", session_id="s1", timestamp=1.0),
                make_smo(item_id="debug-1", item_type="debug", predicate="error", value="500_on_startup", subject="app", session_id="s1", timestamp=1.0),
            ],
            "At work, I prefer Docker for this project.": [
                make_smo(item_id="pref-pos-2", item_type="preference", predicate="pref_pos", value="docker", session_id="s1", timestamp=2.0),
            ],
            "I upgraded to 32GB RAM.": [
                make_smo(item_id="fact-2", item_type="fact", predicate="hw", value="32gb_ram", session_id="s1", timestamp=3.0),
            ],
        }

        def fake_extract(text: str, session_id: str):
            return scripted_turns[text]

        engine.extractor.extract = fake_extract

        reports = engine.ingest_turns(
            [
                "I need to deploy a Flask API tomorrow, prefer nginx, avoid docker, and I am debugging a 500 startup failure on Flask plus Gunicorn with 16GB RAM.",
                "At work, I prefer Docker for this project.",
                "I upgraded to 32GB RAM.",
            ],
            session_id="s1",
            user_id="u1",
            visibility=VISIBILITY_TEAM,
        )

        self.assertFalse(reports[0].has_conflicts)
        self.assertEqual(reports[1].conflicts()[0].resolution, "contextual")
        self.assertEqual(reports[2].conflicts()[0].resolution, "superseded")

        spl, complexity = engine.build_spl(
            "How should I debug and compare deployment options for my Flask API next week?",
            session_id="s1",
            schema_name="coding",
            user_id="u1",
        )
        self.assertEqual(complexity.tier, "planning")
        self.assertEqual(complexity.token_budget, 140)
        self.assertIn("stack=[", spl)
        self.assertIn("flask", spl)
        self.assertIn("gunicorn", spl)
        self.assertIn("err=[500_on_startup]", spl)
        self.assertIn("hw=32gb_ram", spl)
        self.assertNotIn("hw=16gb_ram", spl)
        # Contextual docker preference must appear in ?pref= (scoped), not pref= or !pref=
        self.assertIn("?pref=", spl)
        self.assertIn("docker", spl)
        self.assertNotIn("\npref=[docker", spl)   # not in active pref slot
        self.assertNotIn("!pref=[docker", spl)    # not in active neg-pref slot

        active_memories = engine.store.fetch_active_by_session("s1")
        self.assertTrue(all(item.memory_state == MEMORY_STATE_ACTIVE for item in active_memories))
        self.assertFalse(any(item.value == "docker" for item in active_memories))

        federated_spl, federated_complexity = engine.build_spl(
            "How should I debug this Flask deployment and what stack am I using?",
            session_id="s2",
            schema_name="coding",
            user_id="u1",
        )
        self.assertEqual(federated_complexity.tier, "planning")
        self.assertIn("stack=[", federated_spl)
        self.assertIn("flask", federated_spl)
        self.assertIn("gunicorn", federated_spl)
        self.assertIn("err=[500_on_startup]", federated_spl)
        self.assertIn("task=flask_api", federated_spl)

    def test_real_extractor_populates_v2_coding_slots_in_spl(self):
        config = self.make_test_config(similarity_threshold=0.999)
        engine = SemanticRuntimeEngine(config=config)
        engine.extractor.embedder = HashEmbedder()
        engine.retriever.embedder = HashEmbedder()

        engine.ingest_turns(
            [
                (
                    "I need to deploy a Flask API tomorrow. "
                    "I'm using Flask, Gunicorn, and Nginx, and I keep getting a 500 on startup."
                )
            ],
            session_id="real-text",
            user_id="u1",
            visibility=VISIBILITY_TEAM,
        )

        spl, complexity = engine.build_spl(
            "How should I debug this deployment?",
            session_id="real-text",
            schema_name="coding",
            user_id="u1",
        )

        self.assertEqual(complexity.tier, "planning")
        self.assertIn("stack=[", spl)
        self.assertIn("flask", spl)
        self.assertIn("gunicorn", spl)
        self.assertIn("nginx", spl)
        self.assertIn("500_on_startup", spl)

    def test_default_schema_is_used_when_schema_name_is_omitted(self):
        config = self.make_test_config(
            similarity_threshold=0.999,
            default_schema="coding",
        )
        engine = SemanticRuntimeEngine(config=config)
        engine.extractor.embedder = HashEmbedder()
        engine.retriever.embedder = HashEmbedder()

        engine.ingest_turns(
            [
                "I'm using Flask and Gunicorn and getting a 500 on startup.",
            ],
            session_id="default-schema",
            user_id="u1",
            visibility=VISIBILITY_TEAM,
        )

        spl, _ = engine.build_spl(
            "How should I debug this deployment?",
            session_id="default-schema",
            user_id="u1",
        )

        self.assertIn("stack=[", spl)
        self.assertIn("err=[", spl)

    def test_answer_includes_contextual_preferences_in_prompt(self):
        config = self.make_test_config(similarity_threshold=0.999)
        engine = SemanticRuntimeEngine(config=config)
        engine.extractor.embedder = HashEmbedder()
        engine.retriever.embedder = HashEmbedder()

        engine.ingest_turns(
            [
                "Please avoid docker.",
                "At work, I prefer Docker for this project.",
            ],
            session_id="contextual-answer",
            user_id="u1",
            visibility=VISIBILITY_TEAM,
        )

        captured: dict[str, str] = {}

        def fake_generate(prompt: str, system: str | None = None):
            captured["prompt"] = prompt
            return {"response": "ok"}

        engine.ollama.generate = fake_generate

        result = engine.answer(
            "What should I use?",
            session_id="contextual-answer",
            schema_name="coding",
            user_id="u1",
        )

        self.assertEqual(result["response"], "ok")
        self.assertIn("?pref=[", captured["prompt"])
        self.assertIn("docker", captured["prompt"])
