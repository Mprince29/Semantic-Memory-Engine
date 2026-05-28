import unittest

from semantic_memory.domain.models import SemanticMemoryObject
from semantic_memory.prompting.spl import SPLEncoder
from semantic_memory.spl_schema.registry import SPLSchemaRegistry
from semantic_memory.training.colab import build_colab_training_script


class SPLEncoderTests(unittest.TestCase):
    def test_encode_groups_core_fields(self):
        encoder = SPLEncoder()
        smos = [
            SemanticMemoryObject(id="1", type="task", subject="user", predicate="build", value="ollama project"),
            SemanticMemoryObject(id="2", type="constraint", subject="task", predicate="deadline", value="T+1"),
            SemanticMemoryObject(id="3", type="preference", subject="user", predicate="pref_pos", value="python"),
        ]
        prompt = encoder.encode(smos, "How do I containerize this?")
        self.assertIn("[CTX]", prompt)
        self.assertIn("task=ollama_project", prompt)
        self.assertIn("deadline=T+1", prompt)
        self.assertIn("pref=[python]", prompt)

    def test_schema_registry_exposes_all_v2_domains(self):
        self.assertEqual(
            set(SPLSchemaRegistry.available()),
            {"general", "coding", "medical", "legal"},
        )

    def test_encode_general_schema(self):
        encoder = SPLEncoder()
        smos = [
            SemanticMemoryObject(id="1", type="task", subject="user", predicate="build", value="deploy flask"),
            SemanticMemoryObject(id="2", type="constraint", subject="task", predicate="deadline", value="T+1"),
            SemanticMemoryObject(id="3", type="preference", subject="user", predicate="pref_pos", value="nginx"),
            SemanticMemoryObject(id="4", type="preference", subject="user", predicate="pref_neg", value="docker"),
            SemanticMemoryObject(id="5", type="fact", subject="task", predicate="scope", value="local_only"),
        ]
        prompt = encoder.encode(smos, "What should I do next?", schema_name="general")
        self.assertIn("task=deploy_flask", prompt)
        self.assertIn("deadline=T+1", prompt)
        self.assertIn("pref=[nginx]", prompt)
        self.assertIn("!pref=[docker]", prompt)
        self.assertIn("scope=local_only", prompt)

    def test_encode_coding_schema_extensions(self):
        encoder = SPLEncoder()
        smos = [
            SemanticMemoryObject(id="1", type="task", subject="user", predicate="build", value="deploy flask"),
            SemanticMemoryObject(id="2", type="stack", subject="app", predicate="uses", value="flask"),
            SemanticMemoryObject(id="3", type="stack", subject="app", predicate="uses", value="gunicorn"),
            SemanticMemoryObject(id="4", type="debug", subject="app", predicate="error", value="500_on_startup"),
        ]
        prompt = encoder.encode(smos, "How do I debug this?", schema_name="coding")
        self.assertIn("stack=[flask,gunicorn]", prompt)
        self.assertIn("err=[500_on_startup]", prompt)

    def test_encode_medical_schema_extensions(self):
        encoder = SPLEncoder()
        smos = [
            SemanticMemoryObject(id="1", type="symptom", subject="patient", predicate="reports", value="fever"),
            SemanticMemoryObject(id="2", type="vitals", subject="patient", predicate="bp", value="120_80"),
        ]
        prompt = encoder.encode(smos, "What should I monitor?", schema_name="medical")
        self.assertIn("sym=[fever]", prompt)
        self.assertIn("vitals=[120_80]", prompt)

    def test_encode_legal_schema_extensions(self):
        encoder = SPLEncoder()
        smos = [
            SemanticMemoryObject(id="1", type="jurisdiction", subject="case", predicate="applies", value="california"),
            SemanticMemoryObject(id="2", type="clause", subject="contract", predicate="relevant", value="termination"),
        ]
        prompt = encoder.encode(smos, "What governs this dispute?", schema_name="legal")
        self.assertIn("juris=[california]", prompt)
        self.assertIn("clause=[termination]", prompt)

    def test_unknown_schema_raises_key_error(self):
        encoder = SPLEncoder()
        with self.assertRaises(KeyError):
            encoder.encode([], "Hello?", schema_name="unknown")

    def test_training_script_targets_qwen(self):
        script = build_colab_training_script()
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", script)
        self.assertIn("load_in_4bit=True", script)


if __name__ == "__main__":
    unittest.main()
