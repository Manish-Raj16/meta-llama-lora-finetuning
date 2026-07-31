"""
Prepare dataset for LLM fine-tuning.

Tasks:
1. Load collected CSV.
2. Remove duplicate URLs.
3. Remove empty articles.
4. Remove obvious publisher boilerplate.
5. Save cleaned dataset.
"""

import re
import pandas as pd

INPUT_FILE = "data/historical_positive_news.csv"
OUTPUT_FILE = "data/historical_positive_news_clean.csv"


def clean_article(text: str) -> str:
    """
    Remove repeated publisher footer and normalize whitespace.
    """

    if pd.isna(text):
        return ""

    article = str(text)

    footer_markers = [
        "At the fifteenth position",
        "Compared to some years ago",
        "Creation of special chambers",
        "Receive daily news about public management",
        "Please publish modules in offcanvas position",
        "Doing Business",
        "Trading across borders",
        "Property Registration",
        "public procurement framework",
    ]

    positions = []

    for marker in footer_markers:
        idx = article.find(marker)
        if idx != -1:
            positions.append(idx)

    if positions:
        article = article[:min(positions)]

    article = re.sub(r"\n{3,}", "\n\n", article)
    article = re.sub(r"[ \t]+", " ", article)

    return article.strip()


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["url"])

    # Clean articles
    df["article"] = df["article"].apply(clean_article)

    # Remove empty / tiny articles
    df = df[df["article"].str.len() > 200]

    cleaned_rows = len(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Original rows : {original_rows}")
    print(f"Cleaned rows  : {cleaned_rows}")
    print(f"Removed       : {original_rows - cleaned_rows}")
    print()
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()