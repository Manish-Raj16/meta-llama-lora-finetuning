"""
evaluate.py

Evaluate the fine-tuned Llama model on predefined questions.
"""

import csv
import time

from infer import (
    load_model,
    generate_answer,
)


# =====================================================
# Evaluation Questions
# =====================================================

QUESTIONS = [

    "Who is President Faure Essozimna Gnassingbé?",

    "Explain the recent economic reforms in Togo.",

    "Tell me about TIRSAL.",

    "What are the government's agricultural initiatives?",

    "Summarize the investment policies introduced in Togo.",

    "What infrastructure projects are currently underway in Togo?",

    "Explain Togo's business environment reforms.",

    "How is the government improving the economy?",

    "What is the role of ECOWAS in Togo?",

    "Describe the major development projects in Togo.",

]

OUTPUT_FILE = "evaluation_results.csv"


# =====================================================
# Evaluation
# =====================================================

def evaluate():

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    tokenizer, model = load_model()

    results = []

    for index, question in enumerate(QUESTIONS, start=1):

        print(f"\n[{index}/{len(QUESTIONS)}] {question}")

        start = time.time()

        answer = generate_answer(
            question,
            tokenizer,
            model,
        )

        end = time.time()

        inference_time = round(end - start, 2)

        print("-" * 60)
        print(answer)
        print("-" * 60)
        print(f"Inference Time : {inference_time} sec")

        results.append({

            "Question": question,

            "Answer": answer,

            "Inference Time (sec)": inference_time,

        })

    return results


# =====================================================
# Save Results
# =====================================================

def save_results(results):

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Question",
                "Answer",
                "Inference Time (sec)",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Results saved to {OUTPUT_FILE}")
    print("=" * 60)


# =====================================================
# Main
# =====================================================

def main():

    results = evaluate()

    save_results(results)


if __name__ == "__main__":
    main()


