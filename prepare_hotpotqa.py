import argparse
import json
import os
import random
import logging
from logging_setup import setup_logging
from collections import Counter
import pandas as pd

setup_logging()

def load_raw(path):
    with open(path) as f:
        return json.load(f)


def stratified_sample(data, n, seed=42):
    if not data:
        raise ValueError("The input dataset is empty.")
    if n < 1 or n > len(data):
        raise ValueError(f"Sample size must be between 1 and {len(data)}")
    random.seed(seed)
    groups = {}
    for entry in data:
        key = (entry["type"], entry["level"])
        groups.setdefault(key, []).append(entry)

    total = sum(len(v) for v in groups.values())
    sampled = []
    for key, entries in groups.items():
        k = max(1, round(n * len(entries) / total))
        sampled.extend(random.sample(entries, min(k, len(entries))))

    random.shuffle(sampled)
    if len(sampled) > n:
        sampled = sampled[:n]
    elif len(sampled) < n:
        pool = [e for e in data if e not in sampled]
        sampled.extend(random.sample(pool, n - len(sampled)))

    counts = Counter((e["type"], e["level"]) for e in sampled)
    logging.info(f"Sampled {len(sampled)} questions:")
    for key, count in sorted(counts.items()):
        logging.info(f"  {key[0]}/{key[1]}: {count}")
    return sampled


def build_corpus(sampled):
    docs = {}
    for entry in sampled:
        for title, sentences in entry["context"]:
            if title not in docs:
                docs[title] = " ".join(sentences)
    logging.info(f"Corpus: {len(docs)} documents")
    return docs


def save(sampled, docs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    qa_rows = [{
        "question": e["question"],
        "answer": e["answer"],
        "type": e["type"],
        "level": e["level"],
    } for e in sampled]

    doc_rows = [{"source": title, "content": text} for title, text in docs.items()]

    qa_path = os.path.join(output_dir, "qa.parquet")
    doc_path = os.path.join(output_dir, "documents.parquet")
    pd.DataFrame(qa_rows).to_parquet(qa_path, index=False)
    pd.DataFrame(doc_rows).to_parquet(doc_path, index=False)

    logging.info(f"Saved {len(qa_rows)} QA pairs to {qa_path}")
    logging.info(f"Saved {len(doc_rows)} documents to {doc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="dataset/HotpotQA/data")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(args.input)
    sampled = stratified_sample(raw, args.n, args.seed)
    docs = build_corpus(sampled)
    save(sampled, docs, args.output_dir)