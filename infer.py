"""
infer.py

Inference using the fine-tuned LoRA adapter
"""

import torch
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Load Model
# =====================================================

def load_model():

    print("=" * 60)
    print("Loading Fine-Tuned Model")
    print("=" * 60)

    print(f"Device : {DEVICE}")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer Loaded")

    print("\nLoading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("✓ Base Model Loaded")

    print("\nLoading LoRA Adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.to(DEVICE)
    model.eval()

    print("✓ Adapter Loaded")
    print("=" * 60)

    return tokenizer, model


# =====================================================
# Generate Answer
# =====================================================

def generate_answer(question, tokenizer, model):

    messages = [
        {
            "role": "system",
            "content": (
               "You are an AI assistant specialized in news related to "
"President Faure Essozimna Gnassingbé.\n\n"
"Answer questions using only the knowledge learned during fine-tuning.\n"
"Keep answers factual, concise, and well structured.\n"
"Do not invent facts.\n"
"If you are unsure, clearly say that you do not have enough information."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return answer


# =====================================================
# Main
# =====================================================

def main():

    tokenizer, model = load_model()

    while True:

        question = input("\nQuestion (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break
        start = time.time()

        answer = generate_answer(
            question,
            tokenizer,
            model,
        )
        end = time.time()

        print("\n" + "=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)
        
        print("\nInference Time : {:.2f} seconds".format(end - start))


if __name__ == "__main__":
    main()