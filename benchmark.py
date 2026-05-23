import json
from pathlib import Path

from semantic_memory.domain.models import SemanticMemoryObject
from semantic_memory.extraction.extractor import SemanticExtractor
from semantic_memory.prompting.spl import SPLEncoder
from semantic_memory.retrieval.deduplicator import SemanticDeduplicator


def rough_token_count(text: str) -> int:
    return int(len(text.split()) / 0.75)


def build_semantic_state(
    history: list[str],
    session_id: str,
    extractor: SemanticExtractor | None = None,
    deduplicator: SemanticDeduplicator | None = None,
) -> list[SemanticMemoryObject]:
    extractor = extractor or SemanticExtractor()
    deduplicator = deduplicator or SemanticDeduplicator()
    semantic_state: list[SemanticMemoryObject] = []

    for turn in history:
        new_smos = extractor.extract(turn, session_id=session_id)
        to_add, to_remove = deduplicator.deduplicate(new_smos, semantic_state)
        if to_remove:
            remove_ids = set(to_remove)
            semantic_state = [smo for smo in semantic_state if smo.id not in remove_ids]
        semantic_state = to_add + semantic_state

    return semantic_state


def run_benchmark() -> None:
    base_dir = Path(__file__).resolve().parent
    sample_path = base_dir / "examples" / "sample_conversations.json"
    samples = json.loads(sample_path.read_text())

    encoder = SPLEncoder()

    for sample in samples:
        history = sample["history"]
        query = sample["query"]
        history_text = "\n".join(history)
        full_tokens = rough_token_count(history_text + "\n" + query)

        smos = build_semantic_state(history, session_id=sample["session_id"])
        spl = encoder.encode(smos, query)
        spl_tokens = encoder.count_tokens(spl)
        ratio = round(full_tokens / max(spl_tokens, 1), 2)

        print(f"Session: {sample['session_id']}")
        print(f"Full tokens: {full_tokens}")
        print(f"SPL tokens: {spl_tokens}")
        print(f"Compression ratio: {ratio}x")
        print(spl)
        print("-" * 60)


if __name__ == "__main__":
    run_benchmark()
