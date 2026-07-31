"""
High-Accuracy GPU Sentiment Analysis — DeBERTa-v3-large
==============================================================
Uses microsoft/deberta-v3-large (~400M params) for state-of-the-art accuracy.
Designed specifically for NVIDIA RTX A5000 (24GB VRAM) and similar Ampere GPUs.
Utilizes FP16 Mixed Precision for significantly faster training and lower memory usage.

Run: python train_gpu.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    logging as hf_logging,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

hf_logging.set_verbosity_warning()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = "./data/sentiment_dataset.csv"
SAVE_DIR     = os.environ.get("OWN_MODEL_PATH", "/home/eduadmin/randd/soft_power_positioning/finetuned_model/own_model")
OUTPUT_DIR   = os.path.join(os.path.dirname(SAVE_DIR), "results_gpu")
MAX_LEN      = 128  # Reduced to 128 to ensure it fits alongside Ollama qwen2.5
BATCH_SIZE   = 1    # Extremely low batch size to survive VRAM constraints
GRADIENT_ACCUMULATION_STEPS = 16 # Simulates batch size 16 (1 * 16)
NUM_EPOCHS   = 3
LEARNING_RATE = 5e-5

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k.capitalize() for k, v in LABEL2ID.items()}

import shutil

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
# Do not remove SAVE_DIR here, we will handle it at the end to avoid memory map locks.

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df = df[["text", "label"]].dropna()
df["label"] = df["label"].str.strip().str.lower()
df["label_id"] = df["label"].map(LABEL2ID)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

# Limit dataset if environment variable TRAIN_LIMIT is set
train_limit = os.environ.get("TRAIN_LIMIT")
if train_limit and train_limit.isdigit():
    limit_val = int(train_limit)
    print(f"Limiting dataset to the first {limit_val} rows.")
    df = df.head(limit_val)

print(f"Loaded {len(df)} rows")
print(f"Distribution:\n{df['label'].value_counts().to_string()}\n")

raw = Dataset.from_dict({"text": df["text"].tolist(), "label": df["label_id"].tolist()})
split = raw.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

# ── Tokenize ──────────────────────────────────────────────────────────────────
# Requires sentencepiece installed: pip install sentencepiece protobuf
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['eval'])}")

# ── Model ─────────────────────────────────────────────────────────────────────
# If we are doing a limited run (e.g., 100 rows), we MUST load the previously trained
# model so we don't throw away the classification head and start from random weights!
model_path_to_load = MODEL_NAME
if train_limit and os.path.exists(os.path.join(SAVE_DIR, "config.json")):
    print(f"Incremental training detected! Loading existing model from {SAVE_DIR}")
    model_path_to_load = SAVE_DIR
else:
    print(f"Starting fresh from base model {MODEL_NAME}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_path_to_load, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True
)

# FREEZE BASE MODEL TO PREVENT CATASTROPHIC FORGETTING
print("Freezing base model layers to prevent catastrophic forgetting...")
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False
print("Only the classification head will be trained.")
print(f"Model loaded: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()):,} params)\n")

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4)
    }

# ── Training ──────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True, # Critical for saving memory! Trades computation time for VRAM
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.05,           # Lowered warmup so it starts learning faster
    lr_scheduler_type="cosine",  # Smooth learning rate decay
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=1,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
    bf16=torch.cuda.is_available(),
    fp16=False,
    optim="adamw_torch",
    dataloader_num_workers=0, # Disable multiprocessing to prevent CUDA duplicate memory allocation
)

# Calculate class weights based on the actual distribution of the training set
train_labels = np.array(dataset['train']['label'])
class_counts = np.bincount(train_labels, minlength=3)
total_samples = len(train_labels)
# Inverse frequency weighting
class_weights = total_samples / (3.0 * np.maximum(class_counts, 1))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    compute_metrics=compute_metrics,
)

print(f"Starting training ({MODEL_NAME} on GPU)...")
print("=" * 55)
trainer.train()
print("=" * 55)
print("Training complete!\n")

# ── Save ──────────────────────────────────────────────────────────────────────
import time

TEMP_SAVE_DIR = SAVE_DIR + f"_temp_{int(time.time())}"
trainer.save_model(TEMP_SAVE_DIR)
tokenizer.save_pretrained(TEMP_SAVE_DIR)
print(f"Model successfully saved to {TEMP_SAVE_DIR}")

# Atomically replace SAVE_DIR
if os.path.exists(SAVE_DIR):
    backup_dir = SAVE_DIR + f"_backup_{int(time.time())}"
    os.rename(SAVE_DIR, backup_dir)
    print(f"Moved old model to {backup_dir}")

os.rename(TEMP_SAVE_DIR, SAVE_DIR)
print(f"Model successfully activated at {SAVE_DIR}")

# ── Generate Advanced Evaluation Report & Graphs ──────────────────────────────
print("Running comprehensive evaluation on validation set...")
predictions = trainer.predict(dataset["eval"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
val_loss = predictions.metrics.get('test_loss', 'N/A')

report = {
    "Validation Loss": val_loss,
    "Accuracy": round(acc, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1 Score": round(f1, 4),
    "Perplexity": "N/A (Not applicable for Sequence Classification)",
    "Note": "Metrics calculated using weighted average for imbalanced classes."
}

report_path = os.path.join(SAVE_DIR, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
print(f"Evaluation report saved to {report_path}")

# Plot Learning Rate & Loss Curve
log_history = trainer.state.log_history
steps = []
lrs = []
losses = []

for entry in log_history:
    if 'learning_rate' in entry and 'loss' in entry:
        steps.append(entry['step'])
        lrs.append(entry['learning_rate'])
        losses.append(entry['loss'])

if steps:
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(steps, losses, color='tab:red', label='Training Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Learning Rate', color='tab:blue')  
    ax2.plot(steps, lrs, color='tab:blue', label='Learning Rate', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title("Training Loss and Learning Rate")
    lr_chart_path = os.path.join(SAVE_DIR, "learning_rate_chart.png")
    plt.savefig(lr_chart_path)
    plt.close()
    print(f"Learning rate chart saved to {lr_chart_path}")
else:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "Not enough training steps to generate Learning Rate Curve.\n(Try increasing dataset size or reducing gradient accumulation)", 
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.axis('off')
    lr_chart_path = os.path.join(SAVE_DIR, "learning_rate_chart.png")
    plt.savefig(lr_chart_path)
    plt.close()
    print(f"Learning rate chart (empty) saved to {lr_chart_path}")

# Plot Confusion Matrix
cm = confusion_matrix(labels, preds, labels=list(LABEL2ID.values()))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(LABEL2ID.keys()), yticklabels=list(LABEL2ID.keys()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix graph saved to {cm_path}")

# The evaluation report is already generated earlier in this script.

# ── Inference test ────────────────────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

# Inference on GPU (device=0)
pipe = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

def predict(text: str) -> str:
    result = pipe(text, truncation=True, max_length=MAX_LEN)[0]
    return result["label"].capitalize()

tests = [
    ("This product is absolutely amazing! Best purchase I've ever made.",  "Positive"),
    ("It's okay, nothing special but gets the job done.",                   "Neutral"),
    ("Terrible experience. Would not recommend to anyone.",                 "Negative"),
    ("I love this so much! Exceeded all my expectations!",                  "Positive"),
    ("Complete waste of money. Broke after one day.",                       "Negative"),
]

print("\nInference results:")
print("-" * 65)
correct = 0
for text, expected in tests:
    pred = predict(text)
    match = "✓" if pred.lower() == expected.lower() else "✗"
    correct += (pred.lower() == expected.lower())
    short = text[:55] + "..." if len(text) > 55 else text
    print(f"{match} [{expected:8s}→{pred:8s}]  {short}")
print("-" * 65)
print(f"Accuracy on test samples: {correct}/{len(tests)}")
print(f"\nDone! Model saved in {SAVE_DIR}/")
