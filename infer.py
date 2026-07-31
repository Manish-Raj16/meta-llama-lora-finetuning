"""
Inference Script — Run sentiment predictions after training
===========================================================
Usage:
    # After train_cpu_fast.py (DistilBERT — recommended for CPU):
    python infer.py --model ./model_fast --text "I love this product!"
    python infer.py --model ./model_fast --csv ./data/sentiment_dataset.csv

    # After train.py (Llama LoRA adapter):
    python infer.py --model ./adapter --llama --text "This is terrible."
"""

import argparse
import os

def run_distilbert(model_dir: str, text: str | None, csv_path: str | None):
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-classification", model=model_dir, device=-1)

    if text:
        result = pipe(text, truncation=True, max_length=128)[0]
        print(f"\nText      : {text}")
        print(f"Sentiment : {result['label']} (confidence: {result['score']:.3f})")

    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)
        texts = df["text"].tolist()
        results = pipe(texts, truncation=True, max_length=128, batch_size=16)
        df["predicted"] = [r["label"] for r in results]
        df["confidence"] = [round(r["score"], 3) for r in results]
        out_path = csv_path.replace(".csv", "_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"\nPredictions saved → {out_path}")
        print(df[["text", "label", "predicted", "confidence"]].head(10).to_string())


def run_llama(adapter_dir: str, text: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    from peft import PeftModel

    BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
    SYSTEM_PROMPT = (
        "You are a sentiment analysis expert. "
        "Given a piece of text, classify its sentiment as exactly one of: "
        "Positive, Neutral, or Negative. Reply with only that single word."
    )

    print(f"Loading base model {BASE_MODEL} on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    pipe = hf_pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=10, do_sample=False, repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    )

    def predict(t: str) -> str:
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Classify the sentiment of the following text:\n\n\"{t}\"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        out = pipe(prompt)[0]["generated_text"]
        return out.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()

    if text:
        pred = predict(text)
        print(f"\nText      : {text}")
        print(f"Sentiment : {pred}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model/adapter directory")
    parser.add_argument("--text", default=None, help="Single text to classify")
    parser.add_argument("--csv", default=None, help="CSV file path for batch prediction")
    parser.add_argument("--llama", action="store_true", help="Use Llama LoRA adapter mode")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model directory not found: {args.model}")
        return

    if args.llama:
        run_llama(args.model, args.text)
    else:
        run_distilbert(args.model, args.text, args.csv)


if __name__ == "__main__":
    main()
