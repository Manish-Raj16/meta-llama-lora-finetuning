    """
    Fast CPU Sentiment Analysis — DistilBERT (Recommended for CPU)
    ==============================================================
    Uses DistilBERT (~66M params) instead of Llama 3B.
    Trains in minutes on CPU vs hours.
    Achieves comparable accuracy for sentiment classification.

    Run: python train_cpu_fast.py
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

    hf_logging.set_verbosity_warning()

    # ── Config ────────────────────────────────────────────────────────────────────
    MODEL_NAME   = "distilbert-base-uncased"
    DATASET_PATH = "./data/sentiment_dataset.csv"
    SAVE_DIR     = os.environ.get("OWN_MODEL_PATH", "/home/thulasivl/Python_Project/finetuned_model/own_model")
    OUTPUT_DIR   = os.path.join(os.path.dirname(SAVE_DIR), "results_fast")
    MAX_LEN      = 128
    BATCH_SIZE   = 8
    NUM_EPOCHS   = 3
    LEARNING_RATE = 2e-5

    LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
    ID2LABEL  = {v: k.capitalize() for k, v in LABEL2ID.items()}

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['eval'])}")

    # ── Model ─────────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"Model loaded: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()):,} params)\n")

    # ── Metrics ───────────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": round(float(acc), 4)}

    # ── Training ──────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=5,
        report_to="none",
        use_cpu=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        compute_metrics=compute_metrics,
    )

    print("Starting training (DistilBERT on CPU)...")
    print("=" * 55)
    trainer.train()
    print("=" * 55)
    print("Training complete!\n")

    # ── Save ──────────────────────────────────────────────────────────────────────
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved → {SAVE_DIR}/")

    # ── Inference test ────────────────────────────────────────────────────────────
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

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
    print("\nDone! Model saved in ./model_fast/")
