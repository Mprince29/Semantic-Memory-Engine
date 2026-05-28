import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import build_semantic_state
from semantic_memory.extraction.extractor import SemanticExtractor
from semantic_memory.prompting.builder import SYSTEM_PROMPT
from semantic_memory.prompting.spl import SPLEncoder
from semantic_memory.retrieval.deduplicator import SemanticDeduplicator

random.seed(42)

DROPOUT_INSTRUCTION = (
    "You are a reasoning assistant. Context arrives in Symbolic Prompt Language. "
    "One or more context slots may be absent — answer only from what is provided. "
    "Do not invent missing information."
)

MINIMAL_INSTRUCTION = (
    "You are a reasoning assistant. Context is in compact Symbolic Prompt Language "
    "with only the most relevant slots. Answer precisely from the available slots."
)

DROPPABLE_SLOTS = ["task", "pref", "!pref", "ent", "q_hist", "scope", "hw", "status"]
DOMAIN_SLOT_KEYS = {
    "coding": ("stack=", "err="),
    "medical": ("sym=", "vitals="),
    "legal": ("juris=", "clause="),
}
DEFAULT_OUTPUT_NAMES = (
    "spl_dataset.jsonl",
    "spl_combined.jsonl",
    "spl_final.jsonl",
)


def _rough_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def _extract_slot_values(spl_block: str) -> list[str]:
    ctx_match = re.search(r"\[CTX\](.*?)\[/CTX\]", spl_block, re.DOTALL)
    if not ctx_match:
        return []
    ctx = ctx_match.group(1)
    raw: list[str] = re.findall(r"=([^\s\[\]]+)", ctx)
    raw += re.findall(r"\[([^\]]+)\]", ctx)
    tokens: list[str] = []
    for r in raw:
        for part in r.split(","):
            cleaned = part.strip().replace("_", " ").lower()
            if len(cleaned) > 2:
                tokens.append(cleaned)
    return tokens


def infer_schema_name(smos: list, history: list[str], query: str) -> str:
    types = {smo.type for smo in smos}
    if {"stack", "debug"} & types:
        return "coding"

    text = " ".join(history + [query]).lower()
    medical_keywords = {
        "symptom", "symptoms", "fever", "pain", "dosage", "diagnosis", "patient",
        "medication", "vitals", "clinical",
    }
    legal_keywords = {
        "contract", "clause", "court", "jurisdiction", "statute", "legal",
        "filing", "case law", "plaintiff", "defendant",
    }
    if any(keyword in text for keyword in medical_keywords):
        return "medical"
    if any(keyword in text for keyword in legal_keywords):
        return "legal"
    return "general"


def _slot_coverage(spl_block: str, answer: str) -> float:
    values = _extract_slot_values(spl_block)
    if not values:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for v in values if v in answer_lower)
    return hits / len(values)


def _remove_slot_type(spl_block: str, slot_key: str) -> str:
    lines = spl_block.split("\n")
    filtered = [line for line in lines if not line.strip().startswith(slot_key)]
    return "\n".join(filtered)


def _make_sample(instruction: str, spl_input: str, answer: str) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "input": spl_input,
        "output": answer.strip(),
        "meta": {
            "input_tokens": _rough_tokens(spl_input),
            "output_tokens": _rough_tokens(answer),
            "slot_coverage": round(_slot_coverage(spl_input, answer), 3),
        },
    }


def generate_base_sample(
    history: list[str],
    query: str,
    answer: str,
    session_id: str,
    extractor: SemanticExtractor,
    encoder: SPLEncoder,
    schema_name: str = "general",
) -> dict[str, Any]:
    state = build_semantic_state(history, session_id=session_id, extractor=extractor)
    spl = encoder.encode(state, query, schema_name=schema_name)
    sample = _make_sample(SYSTEM_PROMPT, spl, answer)
    sample["meta"]["schema"] = schema_name
    return sample


def generate_slot_dropout_sample(base_spl: str, query: str, answer: str) -> dict[str, Any] | None:
    present = [s for s in DROPPABLE_SLOTS if f"{s}=" in base_spl]
    if not present:
        return None
    drop_key = random.choice(present)
    trimmed_spl = _remove_slot_type(base_spl, drop_key)
    modified_answer = (
        f"[{drop_key} context not available] " + answer
        if f"{drop_key}" in answer.lower().split()[:6]
        else answer
    )
    sample = _make_sample(DROPOUT_INSTRUCTION, trimmed_spl, modified_answer)
    sample["meta"]["augmentation"] = f"dropout:{drop_key}"
    return sample


def generate_minimal_spl_sample(base_spl: str, query: str, answer: str) -> dict[str, Any] | None:
    ctx_match = re.search(r"(\[CTX\])(.*?)(\[/CTX\])", base_spl, re.DOTALL)
    if not ctx_match:
        return None
    prefix, ctx_body, suffix = ctx_match.group(1), ctx_match.group(2), ctx_match.group(3)
    lines = [ln for ln in ctx_body.strip().split("\n") if ln.strip()]
    priority_keywords = ["task=", "pref=", "!pref=", "scope=", "hw=", "status=", "ent=", "q_hist="]
    ordered: list[str] = []
    for kw in priority_keywords:
        for line in lines:
            if kw in line and line not in ordered:
                ordered.append(line)
                break
    kept = ordered[:2]
    if not kept:
        return None
    query_line = re.search(r"\[Q\].*", base_spl)
    query_part = query_line.group(0) if query_line else f"[Q] {query}"
    minimal_spl = f"{prefix}\n" + "\n".join(kept) + f"\n{suffix}\n{query_part}"
    sample = _make_sample(MINIMAL_INSTRUCTION, minimal_spl, answer)
    sample["meta"]["augmentation"] = "minimal_spl"
    return sample


def generate_schema_extension_sample(base_spl: str, answer: str, schema_name: str) -> dict[str, Any] | None:
    if schema_name == "general":
        return None
    if not any(slot in base_spl for slot in DOMAIN_SLOT_KEYS.get(schema_name, ())):
        return None
    sample = _make_sample(SYSTEM_PROMPT, base_spl, answer)
    sample["meta"]["augmentation"] = f"schema_extension:{schema_name}"
    sample["meta"]["schema"] = schema_name
    return sample


def generate_dataset(source_path: Path, output_path: Path) -> None:
    rows: list[dict] = json.loads(source_path.read_text())
    extractor = SemanticExtractor()
    deduplicator = SemanticDeduplicator()
    encoder = SPLEncoder()

    samples: list[dict[str, Any]] = []
    stats = {
        "total_conversations": len(rows),
        "base_samples": 0,
        "schema_extension_samples": 0,
        "extra_query_samples": 0,
        "slot_dropout_samples": 0,
        "minimal_spl_samples": 0,
        "filtered_low_quality": 0,
    }

    for row in rows:
        session_id = row["session_id"]
        history = row["history"]
        query = row["query"]
        answer = row["answer"]
        extra_queries: list[dict] = row.get("extra_queries", [])
        state = build_semantic_state(history, session_id=session_id, extractor=extractor)
        schema_name = infer_schema_name(state, history, query)

        base = generate_base_sample(
            history,
            query,
            answer,
            session_id,
            extractor,
            encoder,
            schema_name="general",
        )
        base_spl = base["input"]

        if base["meta"]["slot_coverage"] >= 0.05 or len(_extract_slot_values(base_spl)) == 0:
            samples.append(base)
            stats["base_samples"] += 1
        else:
            stats["filtered_low_quality"] += 1

        schema_sample = generate_base_sample(
            history,
            query,
            answer,
            session_id,
            extractor,
            encoder,
            schema_name=schema_name,
        )
        schema_extension = generate_schema_extension_sample(
            schema_sample["input"],
            answer,
            schema_name=schema_name,
        )
        if schema_extension and schema_extension["meta"]["slot_coverage"] >= 0.05:
            samples.append(schema_extension)
            stats["schema_extension_samples"] += 1

        for eq in extra_queries:
            eq_sample = generate_base_sample(
                history,
                eq["query"],
                eq["answer"],
                session_id,
                extractor,
                encoder,
                schema_name=schema_name,
            )
            if eq_sample["meta"]["slot_coverage"] >= 0.05:
                eq_sample["meta"]["augmentation"] = "extra_query"
                eq_sample["meta"]["schema"] = schema_name
                samples.append(eq_sample)
                stats["extra_query_samples"] += 1
            else:
                stats["filtered_low_quality"] += 1

        dropout = generate_slot_dropout_sample(base_spl, query, answer)
        if dropout and dropout["meta"]["slot_coverage"] >= 0.03:
            samples.append(dropout)
            stats["slot_dropout_samples"] += 1

        minimal = generate_minimal_spl_sample(base_spl, query, answer)
        if minimal and minimal["meta"]["slot_coverage"] >= 0.03:
            minimal["meta"]["schema"] = schema_name
            samples.append(minimal)
            stats["minimal_spl_samples"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for filename in DEFAULT_OUTPUT_NAMES:
        target_path = output_path.parent / filename
        with target_path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    stats["total_samples"] = len(samples)
    stats["avg_input_tokens"] = round(
        sum(s["meta"]["input_tokens"] for s in samples) / max(len(samples), 1), 1
    )
    stats["avg_output_tokens"] = round(
        sum(s["meta"]["output_tokens"] for s in samples) / max(len(samples), 1), 1
    )
    stats["avg_slot_coverage"] = round(
        sum(s["meta"]["slot_coverage"] for s in samples) / max(len(samples), 1), 3
    )

    print("=" * 60)
    print("Dataset generation complete")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nOutputs:")
    for filename in DEFAULT_OUTPUT_NAMES:
        print(f"  {output_path.parent / filename}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    generate_dataset(
        source_path=base_dir / "examples" / "sample_conversations.json",
        output_path=base_dir / "finetune" / "data" / "spl_dataset.jsonl",
    )
