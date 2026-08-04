# Fine-Tuning Meta-Llama-3.2-1B-Instruct using LoRA

A domain-specific Large Language Model (LLM) fine-tuning project using **Meta-Llama-3.2-1B-Instruct** and **LoRA (Low-Rank Adaptation)**. The project demonstrates the complete workflow from dataset preparation to model training, inference, and evaluation for domain-specific question answering.

---

## Project Overview

This project fine-tunes the **Meta-Llama-3.2-1B-Instruct** model on a custom instruction dataset built from news articles. Instead of training the entire model, **LoRA (PEFT)** is used to efficiently adapt the model by training only a small set of parameters.

The project includes:

- Dataset preparation
- Instruction dataset generation
- LoRA-based fine-tuning
- Interactive inference
- Model evaluation
- Checkpoint saving and training resume support

---

## Features

- Fine-tuning using LoRA (PEFT)
- Custom instruction dataset
- Hugging Face Transformers integration
- Interactive question-answering
- Automated evaluation script
- Checkpoint-based training recovery
- Modular and reusable code structure

---

## Project Structure

```text
own_model_training/
│
├── data/
│   ├── news.csv
│   └── llama_dataset.jsonl
│
├── prepare_dataset.py
├── convert_dataset.py
├── train_llama.py
├── infer.py
├── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

### Programming

- Python

### Frameworks & Libraries

- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- TRL (SFTTrainer)
- Hugging Face Datasets

### LLM Concepts

- Instruction Fine-Tuning
- Tokenization
- Model Inference
- Model Evaluation

---

## Model

| Property | Value |
|----------|-------|
| Base Model | Meta-Llama-3.2-1B-Instruct |
| Fine-Tuning Method | LoRA (PEFT) |
| Framework | Hugging Face Transformers |
| Trainer | TRL SFTTrainer |

---

## Training Configuration

| Parameter | Value |
|----------|------:|
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | 2e-4 |
| Maximum Sequence Length | 2048 |

---

## LoRA Configuration

| Parameter | Value |
|----------|------:|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | None |

### Target Modules

- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

---

## Workflow

```text
News Dataset
      │
      ▼
Dataset Preparation
      │
      ▼
Instruction Dataset (JSONL)
      │
      ▼
Load Base Llama Model
      │
      ▼
Apply LoRA
      │
      ▼
Tokenization
      │
      ▼
Fine-Tuning
      │
      ▼
Save Adapter
      │
      ▼
Inference
      │
      ▼
Evaluation
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/llm-finetuning-lora.git
cd llm-finetuning-lora
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run the training script

```bash
python train_llama.py
```

The training pipeline performs:

- Load tokenizer
- Load base model
- Apply LoRA
- Load dataset
- Tokenize dataset
- Fine-tune the model
- Save LoRA adapter

---

## Resume Training

If training is interrupted, resume from the latest checkpoint.

Example:

```python
trainer.train(
    resume_from_checkpoint="adapter/checkpoint-280"
)
```

---

## Inference

Run

```bash
python infer.py
```

Example

```
Question:
Tell me about TIRSAL.

Answer:
...
```

---

## Evaluation

Run

```bash
python evaluate.py
```

The evaluation script:

- Loads the fine-tuned model
- Evaluates predefined questions
- Measures inference time
- Saves generated responses for analysis

---

## Results

The fine-tuned model demonstrates domain adaptation by generating responses related to the custom instruction dataset.

Current evaluation focuses on:

- Domain-specific question answering
- Response relevance
- Model inference
- Manual output analysis

---

## Future Improvements

- Improve dataset quality
- Reduce hallucinations through better data curation
- Hyperparameter optimization
- Automatic evaluation metrics
- Retrieval-Augmented Generation (RAG)
- Vector database integration
- Deployment using FastAPI or Flask

---

## Author

**Manish Raj S**

B.Tech – Artificial Intelligence & Data Science
