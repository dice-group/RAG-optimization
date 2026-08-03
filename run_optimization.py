import argparse
import gc
import importlib
import json
import logging
from logging_setup import setup_logging
import pathlib
import time
import faiss
import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, ForbiddenGreaterThanRelation, Integer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from smac import HyperparameterOptimizationFacade, Scenario

from generator import generate
from indexer import load_documents
from pipeline import get_dataset_path, load_qa_pairs
import load_config as config

def build_config_space():
    cs = ConfigurationSpace(seed=42)
    cs.add([Integer("chunk_size", (128, 2048), default=config.chunk_size),
            Integer("chunk_overlap", (0, 200), default=config.chunk_overlap),
            Integer("top_k", (1, 10), default=config.top_k)])
    cs.add(ForbiddenGreaterThanRelation(cs["chunk_overlap"], cs["chunk_size"]))
    return cs

def build_index_for_config(documents, embed_model, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        for split in splitter.split_text(doc["content"]):
            chunks.append({"text": split, "source": doc["source"]})

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(config.embedding_dimension)
    index.add(embeddings)
    return index, chunks

def run_questions(questions, index, chunks, embed_model, top_k, dataset):
    predictions, goldens = [], []
    qa_records = []
    failed_indices = []
    t_search, t_gen = 0.0, 0.0

    for idx, q in enumerate(questions):
        if not q["question"]:
            continue
        try:
            t0 = time.time()
            query_vec = embed_model.encode(
                [q["question"]], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,).astype("float32")
            scores, ids = index.search(query_vec, top_k)
            retrieved = [
                {
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    "score": float(s),
                }
                for s, i in zip(scores[0], ids[0])
                if 0 <= i < len(chunks)
            ]
            t_search += time.time() - t0

            t0 = time.time()
            answer = generate(q["question"], retrieved, dataset)
            t_gen += time.time() - t0

            predictions.append(answer)
            goldens.append(q["golden_answer"])
            qa_records.append({
                "question": q["question"],
                "golden_answer": q["golden_answer"],
                "generated_answer": answer,
                "retrieved_context": "\n\n".join(
                    c["text"] for c in retrieved
                ),
            })
        except Exception as e:
            logging.warning(f"Question {idx} failed: {e}")
            failed_indices.append(idx)
            predictions.append("")
            goldens.append(q["golden_answer"])
            qa_records.append({
                "question": q["question"],
                "golden_answer": q["golden_answer"],
                "generated_answer": "",
                "retrieved_context": "",
            })

    return predictions, goldens, qa_records, failed_indices, t_search, t_gen

def objective(cfg_obj, train_questions, train_negatives, embed_model, documents, state, answers_dir, dataset, method_name, method_module):
    state["trial_count"] += 1
    tid = state["trial_count"]
    cfg = dict(cfg_obj)
    logging.info(f"Trial {tid}: {cfg}")

    if cfg["chunk_overlap"] == cfg["chunk_size"]:
        return 1.0

    t_start = time.time()

    try:
        index, chunks = build_index_for_config(documents, embed_model, cfg["chunk_size"], cfg["chunk_overlap"])
        predictions, goldens, qa_records, failed_indices, t_search, t_gen = run_questions(
            train_questions, index, chunks, embed_model, cfg["top_k"], dataset)

        if not predictions:
            return 1.0

        t_score = time.time()

        if method_name == "bertscore":
            scores = method_module.evaluate_bertscore(predictions, goldens)
            score_key = "avg_bertscore_f1"

        elif method_name == "answer_correctness":
            scores, details = method_module.evaluate_correctness_judge(train_questions, predictions, goldens)
            score_key = "correctness_score"
            for i, record in enumerate(qa_records):
                record["is_correct"] = scores[i] if i < len(scores) else 0.0
                record["correctness_raw"] = details[i]["raw"] if i < len(details) else ""

        elif method_name == "persona":
            scores, details = method_module.evaluate_persona_judges(train_questions, predictions, goldens)
            score_key = "avg_persona_score"
            for i, record in enumerate(qa_records):
                record["persona_avg_score"] = scores[i] if i < len(scores) else 0.0
                record["persona_details"] = details[i] if i < len(details) else {}

        elif method_name == "contrastive":
            scores = method_module.evaluate_contrastive(predictions, goldens, train_negatives, embed_model)
            score_key = "avg_contrastive_score"
            for i, record in enumerate(qa_records):
                record["contrastive_score"] = scores[i]

        else:
            scores = method_module.evaluate_token_recall(predictions, goldens)
            score_key = "avg_token_recall"

        scoring_time = time.time() - t_score
        avg_score = float(np.mean(scores))
        cost = 1.0 - avg_score
        total_time = time.time() - t_start

        trial_data = {
            "trial_id": tid,
            "config": cfg,
            "cost": cost,
            score_key: avg_score,
            "num_failed": len(failed_indices),
            "failed_indices": failed_indices,
            "qa_pairs": qa_records,
        }
        with open(answers_dir / f"trial_{tid}_answers.json", "w") as f:
            json.dump(trial_data, f, indent=2, default=str)

        state["score_key"] = score_key
        state["best_cost"] = min(state["best_cost"], cost)
        state["convergence"].append({
            "trial": tid,
            "cost": cost,
            "best_cost": state["best_cost"],
        })
        state["history"].append({
            "trial_id": tid,
            "config": cfg,
            score_key: avg_score,
            "cost": cost,
            "time_s": round(total_time, 2),
            "search_s": round(t_search, 2),
            "gen_s": round(t_gen, 2),
            "scoring_s": round(scoring_time, 2),
            "num_chunks": len(chunks),
            "num_failed": len(failed_indices),
        })

        logging.info(
            f"Trial {tid}: score={avg_score:.4f} cost={cost:.4f} chunks={len(chunks)} "
            f"failed={len(failed_indices)} "
            f"time={total_time:.1f}s (search={t_search:.1f} gen={t_gen:.1f} score={scoring_time:.1f})")
        return cost

    except torch.cuda.OutOfMemoryError:
        logging.warning(f"Trial {tid} OOM: {cfg}")
        return 1.0

    except Exception as e:
        logging.warning(f"Trial {tid} failed: {cfg} - {e}")
        return 1.0

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_on_test(best_config, test_questions, test_negatives, documents, embed_model, dataset, method_name, method_module):
    cfg = dict(best_config)
    index, chunks = build_index_for_config(documents, embed_model, cfg["chunk_size"], cfg["chunk_overlap"])
    predictions, goldens, qa_records, failed_indices, _, _ = run_questions(
        test_questions, index, chunks, embed_model, cfg["top_k"], dataset)

    if not predictions:
        return 0.0, qa_records, failed_indices

    if method_name == "bertscore":
        scores = method_module.evaluate_bertscore(predictions, goldens)
        for record, score in zip(qa_records, scores):
            record["bertscore_f1"] = score

    elif method_name == "answer_correctness":
        scores, details = method_module.evaluate_correctness_judge(test_questions, predictions, goldens)
        for i, record in enumerate(qa_records):
            record["is_correct"] = scores[i] if i < len(scores) else 0.0
            record["correctness_raw"] = details[i]["raw"] if i < len(details) else ""

    elif method_name == "persona":
        scores, details = method_module.evaluate_persona_judges(test_questions, predictions, goldens)
        for i, record in enumerate(qa_records):
            record["persona_avg_score"] = scores[i] if i < len(scores) else 0.0
            record["persona_details"] = details[i] if i < len(details) else {}

    elif method_name == "contrastive":
        scores = method_module.evaluate_contrastive(predictions, goldens, test_negatives, embed_model)
        for i, record in enumerate(qa_records):
            record["contrastive_score"] = scores[i]

    else:
        scores = method_module.evaluate_token_recall(predictions, goldens)
        for record, score in zip(qa_records, scores):
            record["token_recall"] = score

    return float(np.mean(scores)), qa_records, failed_indices


def run(method_name, dataset_name, n_trials, n_folds, output_dir, negatives_path=None):
    method_module = importlib.import_module(f"optimization_methods.{method_name}")

    if method_name == "bertscore":
        metric = "bertscore_f1"
        train_result_key = "train_f1"
        test_result_key = "test_f1"
        mean_key = "test_f1_mean"
        std_key = "test_f1_std"
    else:
        train_result_key = "train_score"
        test_result_key = "test_score"
        mean_key = "test_score_mean"
        std_key = "test_score_std"
        if method_name == "answer_correctness":
            metric = "correctness_judge"
        elif method_name == "persona":
            metric = "persona_judge_avg"
        elif method_name == "contrastive":
            metric = "contrastive_shifted_sigmoid"
        else:
            metric = "token_recall"

    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = get_dataset_path(dataset_name)
    documents = load_documents(dataset_name, dataset_path)
    qa_pairs = load_qa_pairs(dataset_name, dataset_path)
    embed_model = SentenceTransformer(config.embedding_model)
    all_questions = [{"question": qa["question"], "golden_answer": qa["golden_answer"]} for qa in qa_pairs]

    all_negatives = None
    if method_name == "contrastive":
        all_negatives = method_module.load_negatives(negatives_path)
        logging.info(f"Loaded negatives from {negatives_path}")

    if method_name == "bertscore":
        logging.info(f"BERTScore device: {method_module.bert_device}")
    elif method_name == "persona":
        logging.info(f"Personas: {list(method_module.PERSONAS.keys())}")

    logging.info(f"Loaded {len(all_questions)} questions, {len(documents)} documents from {dataset_name}")
    logging.info(f"Running {n_folds}-fold cross validation with {n_trials} trials per fold")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    t_total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_questions)):
        fold_num = fold_idx + 1
        logging.info(f"Fold {fold_num}/{n_folds}: train={len(train_idx)} test={len(test_idx)}")

        train_questions = [all_questions[i] for i in train_idx]
        test_questions = [all_questions[i] for i in test_idx]
        train_negatives = [all_negatives[i] for i in train_idx] if method_name == "contrastive" else None
        test_negatives = [all_negatives[i] for i in test_idx] if method_name == "contrastive" else None

        fold_dir = out / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        answers_dir = fold_dir / "trial_answers"
        answers_dir.mkdir(parents=True, exist_ok=True)

        with open(fold_dir / "fold_split.json", "w") as f:
            json.dump({"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}, f, indent=2)

        config_space = build_config_space()
        scenario = Scenario(configspace=config_space, deterministic=True, n_trials=n_trials,
                            seed=42, output_directory=fold_dir / "smac_output")

        state = {"trial_count": 0, "best_cost": 1.0, "convergence": [], "history": []}
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=lambda cfg, seed=0, tq=train_questions, tn=train_negatives: objective(
                cfg, tq, tn, embed_model, documents, state, answers_dir,
                dataset_name, method_name, method_module),
            initial_design=HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=10),
            overwrite=True,
        )

        t0 = time.time()
        incumbent = smac.optimize()
        opt_time = time.time() - t0

        if not state["history"]:
            logging.warning(f"Fold {fold_num}: no successful trials, skipping")
            continue

        best_train = min(state["history"], key=lambda x: x["cost"])
        train_score = best_train[state["score_key"]]
        logging.info(f"Fold {fold_num} optimization done in {opt_time:.0f}s, best train score={train_score:.4f}")

        logging.info(f"Fold {fold_num}: evaluating best config on test set")
        test_score, test_records, test_failed = evaluate_on_test(
            incumbent, test_questions, test_negatives, documents, embed_model,
            dataset_name, method_name, method_module)
        logging.info(f"Fold {fold_num} test score={test_score:.4f}, failed={len(test_failed)}")

        best_config = dict(incumbent)
        with open(fold_dir / "test_answers.json", "w") as f:
            json.dump({
                "fold": fold_num,
                "best_config": best_config,
                test_result_key: test_score,
                "num_failed": len(test_failed),
                "failed_indices": test_failed,
                "qa_pairs": test_records,
            }, f, indent=2, default=str)

        with open(fold_dir / "optimization_results.json", "w") as f:
            json.dump({
                "fold": fold_num,
                "best_config": best_config,
                train_result_key: train_score,
                test_result_key: test_score,
                "run_history": state["history"],
                "convergence": state["convergence"],
                "optimization_time_s": round(opt_time, 2),
            }, f, indent=2, default=str)

        fold_results.append({
            "fold": fold_num,
            "best_config": best_config,
            train_result_key: train_score,
            test_result_key: test_score,
            "optimization_time_s": round(opt_time, 2),
        })

    total_time = time.time() - t_total_start

    if not fold_results:
        logging.warning("No folds completed successfully")

    test_scores = [result[test_result_key] for result in fold_results]
    summary = {
        "dataset": dataset_name,
        "n_folds": n_folds,
        "n_trials": n_trials,
        "metric": metric,
        mean_key: float(np.mean(test_scores)) if test_scores else 0.0,
        std_key: float(np.std(test_scores)) if test_scores else 0.0,
        "fold_results": fold_results,
        "total_time_s": round(total_time, 2),
    }
    if method_name == "persona" and fold_results:
        summary["personas"] = list(method_module.PERSONAS.keys())

    with open(out / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info(f"Done in {total_time:.0f}s")

    for result in fold_results:
        logging.info(
            f"  Fold {result['fold']}: "
            f"train={result[train_result_key]:.4f} "
            f"test={result[test_result_key]:.4f} "
            f"config={result['best_config']} "
            f"time={result['optimization_time_s']:.0f}s"
        )

    return summary


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="WikiEval")
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--negatives", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.method == "contrastive" and args.negatives is None:
        parser.error("Negatives is required for the contrastive method")

    run(args.method, args.dataset, args.n_trials, args.n_folds, args.output_dir, args.negatives)