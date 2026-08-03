import logging_setup

import argparse
import json
import logging
import pathlib
import re
import time

import requests

from pipeline import get_dataset_path, load_qa_pairs
import load_config as config

MAX_RETRIES = 3

NEGATIVE_PROMPT = """Given the following question and its correct answer, generate four plausible but incorrect answers.

Question: {question}
Correct answer: {golden_answer}

Requirements:
- Each answer must be factually incorrect
- Each answer must be distinct from the others
- Each answer must be roughly the same length as the correct answer
- Do not include explanations
- Output the four answers as a numbered list: 1. 2. 3. 4."""


def call_llm(prompt, num_predict):
    response = requests.post(
        f"{config.llm_base_url}/api/chat",
        json={
            "model": config.llm_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": {"temperature": 0, "num_predict": num_predict},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_numbered_list(raw):
    parts = re.split(r"\n\s*\d+[\.\)]\s*", "\n" + raw.strip())
    return [p.strip() for p in parts if p.strip()]


def generate_negatives(question, golden_answer, num_predict):
    prompt = NEGATIVE_PROMPT.format(question=question, golden_answer=golden_answer)
    raw = call_llm(prompt, num_predict)
    negatives = parse_numbered_list(raw)

    if len(negatives) == 4:
        return negatives, raw, 1

    for retry in range(MAX_RETRIES):
        logging.warning(f"Retry {retry+1}/{MAX_RETRIES}: got {len(negatives)} negatives, expected 4")
        raw = call_llm(prompt, num_predict)
        negatives = parse_numbered_list(raw)
        if len(negatives) == 4:
            return negatives, raw, retry + 2

    return negatives, raw, MAX_RETRIES + 1


def run(dataset_name, output_path):
    dataset_path = get_dataset_path(dataset_name)
    qa_pairs = load_qa_pairs(dataset_name, dataset_path)
    num_predict = config.num_predict[dataset_name] * 4

    results = []
    for i, qa in enumerate(qa_pairs):
        logging.info(f"[{i+1}/{len(qa_pairs)}] {qa['question'][:80]}...")
        t0 = time.time()
        negatives, raw, attempts = generate_negatives(qa["question"], qa["golden_answer"], num_predict)
        elapsed = time.time() - t0
        results.append({
            "question": qa["question"],
            "golden_answer": qa["golden_answer"],
            "negatives": negatives,
            "raw_response": raw,
            "num_negatives": len(negatives),
            "attempts": attempts,
            "time_seconds": round(elapsed, 2),
        })
        if len(negatives) != 4:
            logging.warning(f"Question {i+1}: gave up after {attempts} attempts, got {len(negatives)} negatives")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    good = sum(1 for r in results if r["num_negatives"] == 4)
    flagged = [i+1 for i, r in enumerate(results) if r["num_negatives"] != 4]
    logging.info(f"Done. {good}/{len(results)} questions got exactly 4 negatives.")
    if flagged:
        logging.warning(f"Flagged questions: {flagged}")
    logging.info(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiEval")
    parser.add_argument("--output", type=str, default="negatives/WikiEval/negatives.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(args.dataset, args.output)