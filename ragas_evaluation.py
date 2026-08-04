import argparse
import asyncio
import json
import logging
from logging_setup import setup_logging
import math
import os
import pathlib
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerCorrectness, SemanticSimilarity

load_dotenv()

methods = ["bertscore", "contrastive", "persona", "naive", "token_recall", "answer_correctness"]
llms = ["gpt", "gemma3", "mistral"]

def results_dir(method, llm):
    if method == "naive":
        return f"naive_rag_results_{llm}"
    return f"optimization_results_{method}_{llm}"

def split_contexts(joined):
    if not joined:
        return []
    return [c for c in joined.split("\n\n") if c.strip()]


def load_fold_samples(test_answers_path):
    with open(test_answers_path) as f:
        data = json.load(f)
    samples = []
    for qa in data["qa_pairs"]:
        if not qa["generated_answer"]:
            continue
        samples.append({
            "user_input": qa["question"],
            "response": qa["generated_answer"],
            "retrieved_contexts": split_contexts(qa["retrieved_context"]),
            "reference": qa["golden_answer"],
        })

    selected_config = data.get("best_config", data.get("config", {}))
    return samples, selected_config


def is_valid(v):
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True

def metric_summary(records):
    cols = ("faithfulness", "answer_correctness", "semantic_similarity")
    means = {}
    stds = {}
    for c in cols:
        vals = [r[c] for r in records if is_valid(r.get(c))]
        n = len(vals)
        means[c] = float(sum(vals) / len(vals)) if vals else None
        if n < 2:
            stds[c] = None
            continue
        m = sum(vals) / n
        stds[c] = float((sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5)
    return means, stds

def score_metric(coro):
    try:
        r = asyncio.run(coro)
        return r.value, None
    except Exception as e:
        return None, str(e)[:200]


def score_one(sample, faithfulness, answer_correctness, semantic_similarity):
    out = {**sample}

    val, err = score_metric(faithfulness.ascore(
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    ))
    out["faithfulness"] = val
    if err:
        out["faithfulness_error"] = err

    val, err = score_metric(answer_correctness.ascore(
        user_input=sample["user_input"],
        response=sample["response"],
        reference=sample["reference"],
    ))
    out["answer_correctness"] = val
    if err:
        out["answer_correctness_error"] = err

    val, err = score_metric(semantic_similarity.ascore(
        response=sample["response"],
        reference=sample["reference"],
    ))
    out["semantic_similarity"] = val
    if err:
        out["semantic_similarity_error"] = err

    return out


def run(method, llm, dataset_name, output_dir):
    method_dir = pathlib.Path(results_dir(method, llm)) / dataset_name
    if not method_dir.exists():
        raise FileNotFoundError(f"No results directory at {method_dir}")

    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
    if not base_url:
        raise SystemExit("Set JUDGE_BASE_URL or OLLAMA_BASE_URL in .env")

    judge_model = os.environ.get("JUDGE_NAME")
    if not judge_model:
        raise SystemExit("Set JUDGE_NAME in .env")

    out = pathlib.Path(output_dir) / method / llm / judge_model / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="not-needed")
    judge = llm_factory(judge_model, client=client, temperature=0.0, max_tokens=2048, extra_body={"reasoning_effort": "none"})
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    faithfulness = Faithfulness(llm=judge)
    answer_correctness = AnswerCorrectness(llm=judge, embeddings=embeddings)
    semantic_similarity = SemanticSimilarity(embeddings=embeddings)

    fold_summaries = []
    for fold_dir in sorted(method_dir.glob("fold_*")):
        fold_num = int(fold_dir.name.split("_")[1])
        test_path = fold_dir / "test_answers.json"
        if not test_path.exists():
            logging.warning(f"Fold {fold_num}: no test_answers.json, skipping")
            continue

        fold_out = out / f"fold_{fold_num}_ragas.json"
        if fold_out.exists():
            with open(fold_out) as f:
                cached = json.load(f)
            fold_summaries.append({
                "fold": fold_num,
                "best_config": cached.get("best_config", {}),
                "n_samples": cached.get("n_samples", 0),
                "means": cached.get("means", {}),
                "stds": cached.get("stds", {}),
            })
            logging.info(f"Fold {fold_num}: cached, skipping")
            continue

        logging.info(f"Fold {fold_num}: loading {test_path}")
        samples, best_config = load_fold_samples(test_path)
        if not samples:
            logging.warning(f"Fold {fold_num}: no valid samples, skipping")
            continue

        logging.info(f"Fold {fold_num}: scoring {len(samples)} samples with RAGAS")
        t0 = time.time()
        records = []
        for i, s in enumerate(samples):
            records.append(score_one(s, faithfulness, answer_correctness, semantic_similarity))
            if (i + 1) % 10 == 0:
                logging.info(f"Fold {fold_num}: {i + 1}/{len(samples)} samples scored")
        elapsed = time.time() - t0

        means, stds = metric_summary(records)

        with open(fold_out, "w") as f:
            json.dump({
                "fold": fold_num,
                "best_config": best_config,
                "n_samples": len(samples),
                "means": means,
                "stds": stds,
                "time_s": round(elapsed, 2),
                "records": records,
            }, f, indent=2, default=str)

        logging.info(f"Fold {fold_num}: {means} in {elapsed:.0f}s")
        fold_summaries.append({
            "fold": fold_num,
            "best_config": best_config,
            "n_samples": len(samples),
            "means": means,
            "stds": stds,
        })

    summary = {
        "method": method,
        "llm": llm,
        "dataset": dataset_name,
        "judge_model": judge_model,
        "fold_results": fold_summaries,
    }
    if fold_summaries:
        metric_names = fold_summaries[0].get("means", {}).keys()
        summary["aggregate"] = {}
        for m in metric_names:
            vals = [f["means"][m] for f in fold_summaries if f.get("means", {}).get(m) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            summary["aggregate"][m] = {"mean": float(mean), "std_across_folds": float(std)}

    with open(out / "ragas_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info(f"Done. Summary: {out / 'ragas_summary.json'}")
    return summary


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=methods)
    parser.add_argument("--llm", required=True, choices=llms)
    parser.add_argument("--dataset", required=True, choices=["WikiEval", "HotpotQA", "RAGMiniBioasq"])
    parser.add_argument("--output_dir", default="ragas_eval_results")
    args = parser.parse_args()
    run(args.method, args.llm, args.dataset, args.output_dir)