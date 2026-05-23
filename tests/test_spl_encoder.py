import unittest

from semantic_memory.domain.models import SemanticMemoryObject
from semantic_memory.prompting.spl import SPLEncoder
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

    def test_training_script_targets_qwen(self):
        script = build_colab_training_script()
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", script)
        self.assertIn("load_in_4bit=True", script)


if __name__ == "__main__":
    unittest.main()
