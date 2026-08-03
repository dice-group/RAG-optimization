import argparse
import os
import random
import logging
from logging_setup import setup_logging
import pandas as pd
from datasets import load_dataset

setup_logging()
def run(output_dir, n, seed):
    qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")["test"]
    corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]

    if n < 1 or n > len(qa):
            raise ValueError(f"Sample size must be between 1 and {len(qa)}")
    
    random.seed(seed)
    indices = random.sample(range(len(qa)), n)
    sampled = [qa[i] for i in indices]

    corpus_map = {int(row["id"]): row["passage"] for row in corpus if row["passage"]}
    docs = {}
    for entry in sampled:
        for pid in entry["relevant_passage_ids"]:
            pid = int(pid)
            if pid in corpus_map:
                docs[pid] = corpus_map[pid]

    os.makedirs(output_dir, exist_ok=True)
    qa_rows = [{"question": e["question"], "answer": e["answer"]} for e in sampled]
    doc_rows = [{"source": str(pid), "content": text} for pid, text in docs.items()]

    pd.DataFrame(qa_rows).to_parquet(os.path.join(output_dir, "qa.parquet"), index=False)
    pd.DataFrame(doc_rows).to_parquet(os.path.join(output_dir, "documents.parquet"), index=False)

    logging.info(f"Sampled {len(qa_rows)} questions, {len(doc_rows)} passages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="dataset/RAGMiniBioasq/data")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output_dir, args.n, args.seed)