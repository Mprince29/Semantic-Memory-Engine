import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import rough_token_count

OLLAMA_URL = "http://localhost:11434/api/generate"
EVAL_SAMPLE_COUNT = 50


def parse_ctx_slots(spl_input: str) -> dict[str, list[str]]:
    ctx_match = re.search(r"\[CTX\](.*?)\[/CTX\]", spl_input, re.DOTALL)
    if not ctx_match:
        return {}
    ctx = ctx_match.group(1)
    slots: dict[str, list[str]] = {}
    for line in ctx.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        kv_match = re.match(r"([!a-z_]+)=(.+)", line)
        if not kv_match:
            continue
        key = kv_match.group(1)
        raw_val = kv_match.group(2).strip()
        if raw_val.startswith("[") and raw_val.endswith("]"):
            values = [v.strip() for v in raw_val[1:-1].split(",") if v.strip()]
        else:
            values = [raw_val]
        slots[key] = [v.replace("_", " ").lower() for v in values]
    return slots


def extract_query(spl_input: str) -> str:
    q_match = re.search(r"\[Q\]\s*(.+)", spl_input)
    return q_match.group(1).strip() if q_match else ""


def all_slot_values(slots: dict[str, list[str]]) -> list[str]:
    values = []
    for vals in slots.values():
        values.extend(vals)
    return values


def compute_slot_recall(slots: dict[str, list[str]], answer: str) -> float:
    values = all_slot_values(slots)
    if not values:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for v in values if v in answer_lower)
    return round(hits / len(values), 3)


def compute_hallucination_score(slots: dict[str, list[str]], query: str, answer: str) -> float:
    allowed_tokens: set[str] = set()
    for v in all_slot_values(slots):
        allowed_tokens.update(v.split())
    allowed_tokens.update(query.lower().split())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "out", "off", "over", "under",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "no", "only", "own", "same", "than", "too", "very", "just",
        "this", "that", "these", "those", "it", "its", "your", "our", "their",
        "you", "we", "they", "he", "she", "i", "me", "my", "his", "her",
        "us", "them", "what", "which", "who", "how", "when", "where", "why",
        "each", "all", "any", "few", "more", "most", "some", "such",
        "if", "then", "because", "as", "since", "while", "although", "unless",
    }
    answer_words = [w.strip(".,!?:;\"'()") for w in answer.lower().split() if len(w) >= 4]
    content_words = [w for w in answer_words if w not in stopwords]
    if not content_words:
        return 0.0
    novel = sum(1 for w in content_words if w not in allowed_tokens)
    return round(novel / len(content_words), 3)


def compute_answer_relevance(query: str, answer: str) -> float:
    q_words = set(query.lower().split()) - {"the", "a", "an", "is", "how", "what", "why", "do", "i"}
    if not q_words:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for w in q_words if w in answer_lower)
    return round(hits / len(q_words), 3)


ALPACA_PROMPT = """\
### Instruction
{instruction}

### Input
{input}

### Response
"""


def query_ollama(model: str, instruction: str, spl_input: str, timeout: int = 60) -> str:
    prompt = ALPACA_PROMPT.format(instruction=instruction, input=spl_input)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        return f"[ERROR: {e}]"


def load_eval_samples(dataset_path: Path, n: int = EVAL_SAMPLE_COUNT) -> list[dict]:
    lines = dataset_path.read_text().splitlines()
    samples = [json.loads(l) for l in lines if l.strip()]
    start = max(0, len(samples) - n)
    return samples[start:]


def evaluate_model(model: str, samples: list[dict], label: str = "") -> dict[str, Any]:
    results = []
    label_str = f"[{label}] " if label else ""
    print(f"\n{label_str}Evaluating {model} on {len(samples)} samples...")

    for i, sample in enumerate(samples, 1):
        spl_input = sample["input"]
        instruction = sample["instruction"]

        slots = parse_ctx_slots(spl_input)
        query = extract_query(spl_input)

        t0 = time.time()
        answer = query_ollama(model, instruction, spl_input)
        latency = round(time.time() - t0, 2)

        results.append({
            "sample_id": i,
            "slot_recall": compute_slot_recall(slots, answer),
            "hallucination": compute_hallucination_score(slots, query, answer),
            "answer_relevance": compute_answer_relevance(query, answer),
            "input_tokens": rough_token_count(spl_input),
            "output_tokens": rough_token_count(answer),
            "latency_s": latency,
        })

        if i % 10 == 0:
            avg_sr = sum(r["slot_recall"] for r in results) / len(results)
            print(f"  {i}/{len(samples)} — avg slot_recall so far: {avg_sr:.3f}")

    avg = lambda key: round(sum(r[key] for r in results) / len(results), 3)
    return {
        "model": model,
        "n_samples": len(results),
        "avg_slot_recall": avg("slot_recall"),
        "avg_hallucination": avg("hallucination"),
        "avg_answer_relevance": avg("answer_relevance"),
        "avg_input_tokens": avg("input_tokens"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_latency_s": avg("latency_s"),
        "results": results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{'='*55}")
    print(f"  Model            : {summary['model']}")
    print(f"  Samples evaluated: {summary['n_samples']}")
    print(f"{'='*55}")
    print(f"  slot_recall      : {summary['avg_slot_recall']:.3f}  (higher = better, target >0.35)")
    print(f"  hallucination    : {summary['avg_hallucination']:.3f}  (lower  = better, target <0.40)")
    print(f"  answer_relevance : {summary['avg_answer_relevance']:.3f}  (higher = better, target >0.50)")
    print(f"  avg input tokens : {summary['avg_input_tokens']}")
    print(f"  avg output tokens: {summary['avg_output_tokens']}")
    print(f"  avg latency      : {summary['avg_latency_s']}s")


def compare_models(base: str, finetuned: str, samples: list[dict]) -> None:
    base_summary = evaluate_model(base, samples, label="BASE")
    ft_summary = evaluate_model(finetuned, samples, label="FINE-TUNED")

    print_summary(base_summary)
    print_summary(ft_summary)

    print(f"\n{'='*55}")
    print("  Delta (fine-tuned vs base)")
    print(f"{'='*55}")

    def delta(key: str) -> str:
        d = ft_summary[key] - base_summary[key]
        sign = "+" if d >= 0 else ""
        better = (d > 0) if "recall" in key or "relevance" in key else (d < 0)
        tag = "✓" if better else "✗"
        return f"{sign}{d:.3f} {tag}"

    print(f"  slot_recall      : {delta('avg_slot_recall')}")
    print(f"  hallucination    : {delta('avg_hallucination')}")
    print(f"  answer_relevance : {delta('avg_answer_relevance')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--dataset", default="finetune/data/spl_final.jsonl")
    parser.add_argument("--n", type=int, default=EVAL_SAMPLE_COUNT)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--base", default="qwen2.5:3b")
    parser.add_argument("--finetuned", default="qwen-spl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = ROOT_DIR / args.dataset
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    samples = load_eval_samples(dataset_path, n=args.n)
    print(f"Loaded {len(samples)} eval samples from {dataset_path.name}")

    if args.compare:
        compare_models(args.base, args.finetuned, samples)
    else:
        summary = evaluate_model(args.model, samples)
        print_summary(summary)
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2))
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
