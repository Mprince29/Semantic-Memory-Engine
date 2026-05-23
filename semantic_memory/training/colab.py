from pathlib import Path

from semantic_memory.config import DEFAULT_CONFIG


def build_colab_training_script() -> str:
    dataset_path = Path("finetune/data/spl_combined.jsonl")
    output_dir = Path(DEFAULT_CONFIG.fine_tune_output_dir)
    model_id = DEFAULT_CONFIG.fine_tune_model_id
    max_seq_length = DEFAULT_CONFIG.fine_tune_max_seq_length
    return f"""# ============================================================
# SPL Fine-Tuning Script — Qwen2.5-3B-Instruct via QLoRA
# Run this cell-by-cell in Google Colab (T4 GPU runtime)
# ============================================================

# ----- Cell 1: Install dependencies -----
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes datasets

# ----- Cell 2: Upload dataset -----
# Upload finetune/data/spl_combined.jsonl via the Colab file browser (left sidebar)
# or run this to mount Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_PATH = "/content/drive/MyDrive/spl_combined.jsonl"
DATASET_PATH = "/content/spl_combined.jsonl"

# ----- Cell 3: Load model -----
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_id}",
    max_seq_length={max_seq_length},
    dtype=None,          # auto-detect: bf16 on Ampere, fp16 on T4
    load_in_4bit=True,
)

# ----- Cell 4: Attach LoRA adapters -----
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ----- Cell 5: Prepare dataset -----
from datasets import load_dataset

ALPACA_PROMPT = \"\"\"### Instruction
{{instruction}}

### Input
{{input}}

### Response
{{output}}{{eos}}\"\"\"

EOS = tokenizer.eos_token

def format_sample(row):
    return {{
        "text": ALPACA_PROMPT.format(
            instruction=row["instruction"],
            input=row["input"],
            output=row["output"],
            eos=EOS,
        )
    }}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# 90/10 train/eval split
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train samples : {{len(train_dataset)}}")
print(f"Eval samples  : {{len(eval_dataset)}}")
print("\\nSample prompt preview:")
print(train_dataset[0]["text"][:400])

# ----- Cell 6: Train -----
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length={max_seq_length},
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch = 16
        num_train_epochs=5,
        warmup_ratio=0.1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        output_dir="{output_dir}",
        report_to="none",
        seed=42,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU: {{gpu_stats.name}}  VRAM: {{gpu_stats.total_memory / 1e9:.1f}} GB")

trainer_stats = trainer.train()
print(f"\\nTraining complete in {{trainer_stats.metrics['train_runtime']:.0f}}s")
print(f"Final train loss: {{trainer_stats.metrics['train_loss']:.4f}}")

# ----- Cell 7: Quick inference test -----
FastLanguageModel.for_inference(model)

TEST_PROMPTS = [
    \"\"\"### Instruction
You are a reasoning assistant. Context arrives in Symbolic Prompt Language.
Read the structure directly. Keys may include task, deadline, pref, !pref, ent, q_hist, scope, hw, and status.
Respond precisely and do not invent missing facts.

### Input
[CTX]
task=deploy_flask_app deadline=T+1
pref=[nginx,systemd]
!pref=[docker]
scope=self_hosted hw=2gb_ram
q_hist=[flask,gunicorn,ubuntu]
[/CTX]
[Q] How do I keep the app running after reboot?

### Response
\"\"\",
    \"\"\"### Instruction
You are a reasoning assistant. Context arrives in Symbolic Prompt Language.
Read the structure directly. Keys may include task, deadline, pref, !pref, ent, q_hist, scope, hw, and status.
Respond precisely and do not invent missing facts.

### Input
[CTX]
task=reduce_api_costs
pref=[structured_context,minimal_tokens]
!pref=[cloud_inference]
status=in_progress
q_hist=[token,compression,llm,context]
[/CTX]
[Q] What is the fastest way to cut token usage by 60%?

### Response
\"\"\",
]

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\\n=== Test {{i}} ===")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
    )

# ----- Cell 8: Save -----
model.save_pretrained("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print(f"\\nModel saved to: {output_dir}")

# To save to Google Drive:
# model.save_pretrained("/content/drive/MyDrive/qwen-spl-lora")
# tokenizer.save_pretrained("/content/drive/MyDrive/qwen-spl-lora")

# To export as GGUF for Ollama (optional):
# model.save_pretrained_gguf("{output_dir}-gguf", tokenizer, quantization_method="q4_k_m")
""".strip()
