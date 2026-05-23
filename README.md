# Semantic Memory Engine

Backend-only semantic memory and prompt compression engine for local LLMs running through Ollama.

## The Problem

Most prompt compression systems still waste tokens because they compress context but then reconstruct verbose natural language before sending it to the model — paying the token cost twice.

## The Approach

Conversations are converted into **Symbolic Prompt Language (SPL)** — a compact slot-value format — and a small local model is fine-tuned to reason from it directly, with no natural language reconstruction in the middle.

```
conversation → semantic extraction → SPL encoding → fine-tuned local model → answer
```

SPL example:
```
[CTX]
task=deploy_flask_app deadline=T+1
pref=[nginx,systemd]
!pref=[docker,managed_cloud]
hw=2gb_ram scope=self_hosted
q_hist=[flask,gunicorn,ubuntu]
[/CTX]
[Q] How do I keep the app running after reboot?
```

This replaces ~1200 tokens of conversation history with ~20 tokens of structured context — roughly 10x compression.

## Results

Fine-tuning Qwen2.5-3B on 376 SPL instruction-response pairs via QLoRA:

| Metric | Base Qwen2.5-3B | Fine-tuned | Delta |
|---|---|---|---|
| slot_recall | 0.000 | 0.556 | +0.556 |
| answer_relevance | 0.000 | 0.610 | +0.610 |
| hallucination | 1.000 | 0.738 | -0.262 |
| avg input tokens | — | 18 | ~10x compression vs full history |

The base model scored zero — it had no concept of SPL. The fine-tuned model reads and reasons from the compressed format natively.

## Architecture

```
semantic-memory-engine/
├── semantic_memory/
│   ├── application/engine.py        # main pipeline entry point
│   ├── config.py                    # engine configuration
│   ├── domain/models.py             # SemanticMemoryObject
│   ├── extraction/extractor.py      # conversation → semantic units
│   ├── infrastructure/
│   │   ├── ollama.py                # Ollama API client
│   │   └── store.py                 # SQLite + ChromaDB persistence
│   ├── prompting/
│   │   ├── builder.py               # prompt assembly
│   │   └── spl.py                   # SPL encoder
│   ├── retrieval/
│   │   ├── deduplicator.py          # semantic deduplication
│   │   └── retriever.py             # query-aware retrieval with token budget
│   └── training/colab.py            # Colab training script builder
├── finetune/
│   ├── dataset_generator.py         # generates SPL training data with augmentation
│   ├── eval_compression.py          # slot-recall evaluation against Ollama models
│   ├── train_lora.py                # prints the Colab training script
│   ├── train_spl.ipynb              # Colab notebook (T4 GPU, QLoRA)
│   └── data/                        # training datasets (gitignored)
├── examples/sample_conversations.json
├── tests/test_spl_encoder.py
└── benchmark.py
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with `qwen2.5:3b` pulled:
```bash
ollama pull qwen2.5:3b
```

## Usage

Run the demo pipeline (extract → SPL → Ollama):
```bash
python engine.py
```

Run the compression benchmark:
```bash
python benchmark.py
```

Generate the SPL training dataset:
```bash
python finetune/dataset_generator.py
```

Evaluate a model's SPL reasoning (requires Ollama):
```bash
python finetune/eval_compression.py --model qwen-spl

# Compare base vs fine-tuned
python finetune/eval_compression.py --compare --base qwen2.5:3b --finetuned qwen-spl
```

## Fine-Tuning

Generate the Colab training script:
```bash
python finetune/train_lora.py
```

Or open `finetune/train_spl.ipynb` directly in Google Colab (T4 GPU runtime). Upload `finetune/data/spl_final.jsonl` and run all cells. Training takes ~10 minutes on T4 for 376 samples.

After training, export the model as GGUF and load into Ollama:
```bash
ollama create qwen-spl -f artifacts/Modelfile
```

**Pre-trained GGUF model** (Q4_K_M, ~2GB) is available on Google Drive — too large for GitHub:
[qwen2.5-3b-instruct.Q4_K_M.gguf](https://drive.google.com/drive/folders/1utQm3tAA-DTFyaZIrL-05dAX2kz2M1Nz?usp=sharing)

Download it, place it in `artifacts/`, then run:
```bash
ollama create qwen-spl -f artifacts/Modelfile
ollama run qwen-spl
```

## Runtime Flow

1. Extract semantic units from conversation turns (tasks, preferences, constraints, entities, events)
2. Embed and deduplicate overlapping meaning
3. Persist in SQLite and ChromaDB
4. Retrieve query-relevant memories under a token budget
5. Encode as SPL
6. Send to fine-tuned local model via Ollama

## Stack

- Python 3.11+
- [Ollama](https://ollama.com) — local model inference
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — base model
- QLoRA via [Unsloth](https://github.com/unslothai/unsloth) + PEFT
- ChromaDB — vector store
- SQLite — structured memory persistence
- sentence-transformers — embeddings
- spaCy — NER extraction
