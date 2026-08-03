import logging
import os
import json
import time
import glob
import pyarrow.parquet as parquet
from bert_score import score as bert_score
from indexer import run as run_indexer
from retriever import load_index, retrieve
from generator import generate

def get_dataset_path(dataset_name):
    pattern = f"dataset/{dataset_name}/data/*.parquet"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files found in dataset/{dataset_name}/data/")
    qa_files = [f for f in files if f.endswith("qa.parquet")]
    if qa_files:
        return qa_files[0]
    return files[0]

def load_qa_pairs(dataset_name, dataset_path):
    if dataset_name not in ("WikiEval", "HotpotQA", "RAGMiniBioasq"):
        raise ValueError(f"No QA loader for '{dataset_name}'")
    table = parquet.read_table(dataset_path)
    return [
                {"question": table.column("question")[i].as_py(),
                 "golden_answer": table.column("answer")[i].as_py()}
                for i in range(table.num_rows)
            ]


def evaluate(results):
    if not results:
        raise ValueError("No results available for evaluation.")
    generated = [r["generated_answer"] for r in results]
    golden = [r["golden_answer"] for r in results]
    P, R, F1 = bert_score(generated, golden, lang="en", verbose=False)

    for i, r in enumerate(results):
        r["bertscore_precision"] = float(P[i])
        r["bertscore_recall"] = float(R[i])
        r["bertscore_f1"] = float(F1[i])

    avg_f1 = float(F1.mean())
    total_time = sum(r["time_seconds"] for r in results)
    avg_time = total_time / len(results)
    
    return {"avg_bertscore_f1": avg_f1, "avg_latency": avg_time, "total_time": total_time}


def run(dataset_name):
    dataset_path = get_dataset_path(dataset_name)
    index_dir = f"faiss_index/{dataset_name}"
    results_dir = f"results/{dataset_name}"

    run_indexer(dataset_name, dataset_path, index_dir)
    index, chunks = load_index(index_dir)
    qa_pairs = load_qa_pairs(dataset_name, dataset_path)

    results = []
    for i, qa in enumerate(qa_pairs):
        logging.info(f"[{i+1}/{len(qa_pairs)}] {qa['question'][:80]}...")
        start = time.time()
        retrieved = retrieve(qa["question"], index, chunks)
        answer = generate(qa["question"], retrieved, dataset_name)
        elapsed = time.time() - start

        results.append({
            "question": qa["question"],
            "golden_answer": qa["golden_answer"],
            "generated_answer": answer,
            "retrieved_chunks": [c["text"] for c in retrieved],
            "retrieval_scores": [c["score"] for c in retrieved],
            "time_seconds": round(elapsed, 2),
        })

    summary = evaluate(results)

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run(args.dataset)