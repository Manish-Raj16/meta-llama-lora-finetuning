"""
train_llama.py

Fine-tuning Meta-Llama-3.2-1B-Instruct using LoRA

Pipeline
--------
1. Load tokenizer
2. Load base model
3. Apply LoRA
4. Load dataset
5. Tokenize dataset
6. Configure trainer
7. Fine-tune model
8. Save adapter
9. Test inference
"""

import os
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers import set_seed
# set_seed(42)
from trl import SFTTrainer

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from datasets import load_dataset
# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

OUTPUT_DIR = "adapter"

NUM_EPOCHS = 3

BATCH_SIZE = 1

LEARNING_RATE = 2e-4

GRADIENT_ACCUMULATION = 8

DATASET_PATH = "data/llama_dataset.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42)
print("=" * 60)
print("LLM Fine-Tuning Project")
print("=" * 60)

print(f"Model          : {MODEL_NAME}")
print(f"Device         : {DEVICE}")
print(f"PyTorch        : {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

print("=" * 60)


# =====================================================
# Load Tokenizer
# =====================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Tokenizer loaded successfully")


# =====================================================
# Load Base Model
# =====================================================

print("\nLoading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype=torch.float32,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False

print("✓ Base model loaded successfully")

# =====================================================
# Apply LoRA
# =====================================================

print("\nApplying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

print("✓ LoRA applied successfully")

# =====================================================
# Model Summary
# =====================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel Summary")
print("=" * 60)

print(f"Total Parameters      : {total_params:,}")
print(f"Trainable Parameters  : {trainable_params:,}")

print(
    f"Trainable Percentage  : "
    f"{100 * trainable_params / total_params:.4f}%"
)

print("=" * 60)

print("\nMilestone 2 Completed Successfully!")
# =====================================================
# Load Dataset
# =====================================================

print("\nLoading training dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print("✓ Dataset loaded successfully")

print("\nDataset Summary")
print("=" * 60)

print(f"Number of samples : {len(dataset)}")
print(f"Columns           : {dataset.column_names}")

print("=" * 60)

print("\nFirst Sample")
print("=" * 60)

print(dataset[0])

print("=" * 60)


print("\nMilestone 3 Completed Successfully!")

# =====================================================
# Tokenization
# =====================================================

print("\nTokenizing dataset...")

def format_chat(example):

    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_function(example):

    text = format_chat(example)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        # padding="max_length",
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
)

print("✓ Dataset tokenized successfully")

# =====================================================
# Tokenized Dataset Summary
# =====================================================

print("\nTokenized Dataset Summary")
print("=" * 60)

print(tokenized_dataset)

print("=" * 60)

print("\nFirst Tokenized Sample")
print("=" * 60)

print(tokenized_dataset[0])

print("=" * 60)

print("\nMilestone 4 Completed Successfully!")

# =====================================================
# Training Arguments
# =====================================================

print("\nCreating Training Arguments...")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    logging_steps=10,

    save_strategy="epoch",

    save_total_limit=2,

    fp16=False,

    bf16=False,

    report_to="none",
    remove_unused_columns=False,

)

print("✓ Training Arguments Created")

# =====================================================
# Data Collator
# =====================================================

print("\nCreating Data Collator...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

print("✓ Data Collator Created")

# =====================================================
# Trainer
# =====================================================

print("\nCreating Trainer...")

trainer = SFTTrainer(

    model=model,

    train_dataset=tokenized_dataset,

    args=training_args,

    data_collator=data_collator,

)

print("✓ Trainer Created Successfully")
print("\nMilestone 5 Completed Successfully!")

print("\nTraining Configuration")
print("=" * 60)

print(f"Epochs                 : {NUM_EPOCHS}")
print(f"Batch Size             : {BATCH_SIZE}")
print(f"Gradient Accumulation  : {GRADIENT_ACCUMULATION}")
print(f"Learning Rate          : {LEARNING_RATE}")
print(f"Dataset Size           : {len(tokenized_dataset)}")

print("=" * 60)
print(f"\nAdapter will be saved to: {OUTPUT_DIR}")
start_time = time.time()

# =====================================================
# Fine-Tuning
# =====================================================

print("\nStarting Fine-Tuning...")
print("=" * 60)

start_time = time.time()

trainer.train()

end_time = time.time()

print("=" * 60)
print("Fine-Tuning Completed Successfully!")
print("=" * 60)

print(f"Training Time : {(end_time - start_time)/60:.2f} minutes")

# =====================================================
# Save Adapter
# =====================================================

print("\nSaving LoRA Adapter...")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

trainer.save_state()

print("✓ LoRA Adapter Saved Successfully")
print(f"Saved Location : {OUTPUT_DIR}")

print("=" * 60)

print("\nMilestone 7 Completed Successfully!")