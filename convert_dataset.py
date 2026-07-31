"""
Convert cleaned president news dataset into Llama instruction dataset.
"""

import json
import pandas as pd

INPUT_FILE = "data/historical_positive_news_clean.csv"
OUTPUT_FILE = "data/llama_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are an AI assistant specialized in news related to "
    "President Faure Essozimna Gnassingbé. "
    "Answer questions accurately using the information you have learned."
)


def main():

    print("=" * 60)
    print("Converting dataset...")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    total_rows = len(df)
    converted = 0
    skipped = 0

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for _, row in df.iterrows():

            title = str(row.get("title", "")).strip()
            article = str(row.get("article", "")).strip()

            if len(title) == 0 or len(article) == 0:
                skipped += 1
                continue

            training_examples = [
    {
        "user": f"Tell me about: {title}",
        "assistant": article,
    },
    {
        "user": f"What happened regarding: {title}?",
        "assistant": article,
    },
    {
        "user": f"Explain the following news topic: {title}",
        "assistant": article,
    },
    {
        "user": f"Provide detailed information about: {title}",
        "assistant": article,
    },
    {
        "user": f"Summarize the news titled: {title}",
        "assistant": article,
    },
    {
        "user": f"What are the key highlights of: {title}?",
        "assistant": article,
    },
    {
        "user": f"Describe the main event discussed in: {title}",
        "assistant": article,
    },
    {
        "user": f"Give an overview of: {title}",
        "assistant": article,
    },
]

            for example in training_examples:
            
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": example["user"],
                        },
                        {
                            "role": "assistant",
                            "content": example["assistant"],
                        },
                    ]
                }
            
                fout.write(
                    json.dumps(sample, ensure_ascii=False) + "\n"
                )
            
                converted += 1
    print()
    print("=" * 60)
    print("Dataset Conversion Completed")
    print("=" * 60)

    print(f"Rows Read      : {total_rows}")
    print(f"Rows Converted : {converted}")
    print(f"Rows Skipped   : {skipped}")

    print()
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()