import unittest

from finetune.dataset_generator import infer_schema_name
from finetune.eval_compression import compute_extension_slot_recall
from semantic_memory.domain.models import SemanticMemoryObject


class FineTuneV2Tests(unittest.TestCase):
    def test_infer_schema_name_prefers_coding_when_stack_or_debug_present(self):
        smos = [
            SemanticMemoryObject(
                id="1",
                type="stack",
                subject="app",
                predicate="uses",
                value="flask",
            ),
            SemanticMemoryObject(
                id="2",
                type="debug",
                subject="app",
                predicate="error",
                value="500_on_startup",
            ),
        ]

        schema = infer_schema_name(
            smos,
            history=["I'm using Flask and getting a 500 on startup."],
            query="How should I debug this deployment?",
        )

        self.assertEqual(schema, "coding")

    def test_compute_extension_slot_recall_only_scores_v2_slots(self):
        slots = {
            "task": ["deploy flask app"],
            "stack": ["flask", "gunicorn"],
            "err": ["500 on startup"],
        }
        answer = "Debug the Flask app by checking Gunicorn logs for the 500 on startup issue."

        self.assertEqual(compute_extension_slot_recall(slots, answer), 1.0)


if __name__ == "__main__":
    unittest.main()
