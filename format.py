input_file = "./data/sentiment_dataset.csv"
output_file = "./data/sentiment_dataset_fixed.csv"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    fout.write("text,label,sentiment\n")

    next(fin)  # skip header

    for line in fin:
        line = line.strip()

        if not line:
            continue

        text, label, sentiment = line.rsplit(",", 2)

        text = text.replace('"', '""')

        fout.write(f'"{text}",{label},{sentiment}\n')

print(f"Fixed dataset saved to {output_file}")