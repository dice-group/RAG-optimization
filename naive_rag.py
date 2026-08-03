import gc
import json
import logging
from logging_setup import setup_logging
import pathlib
import time
import torch
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import KFold
 
from pipeline import get_dataset_path, load_qa_pairs
from indexer import load_documents
from generator import generate
import load_config as config
 
naive_config = {
    "chunk_size": config.chunk_size,
    "chunk_overlap": config.chunk_overlap,
    "top_k": config.top_k,
}
 
 
def build_index(documents, embed_model, chunk_size, chunk_overlap):
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
                convert_to_numpy=True, show_progress_bar=False,
            ).astype("float32")
            scores, ids = index.search(query_vec, top_k)
            retrieved = [
                {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if 0 <= i < len(chunks)
            ]
            t_search += time.time() - t0
 
            t0 = time.time()
            answer = generate(q["question"], retrieved, dataset)
            t_gen += time.time() - t0
 
            qa_records.append({
                "question": q["question"],
                "golden_answer": q["golden_answer"],
                "generated_answer": answer,
                "retrieved_context": "\n\n".join(c["text"] for c in retrieved),
            })
        except Exception as e:
            logging.warning(f"Question {idx} failed: {e}")
            failed_indices.append(idx)
            qa_records.append({
                "question": q["question"],
                "golden_answer": q["golden_answer"],
                "generated_answer": "",
                "retrieved_context": "",
            })
 
    return qa_records, failed_indices, t_search, t_gen
 
 
def run(dataset_name, n_folds, output_dir):
    out = pathlib.Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
 
    dataset_path = get_dataset_path(dataset_name)
    documents = load_documents(dataset_name, dataset_path)
    qa_pairs = load_qa_pairs(dataset_name, dataset_path)
    embed_model = SentenceTransformer(config.embedding_model)
 
    all_questions = [{"question": qa["question"], "golden_answer": qa["golden_answer"]}
                     for qa in qa_pairs]
 
    logging.info(f"Loaded {len(all_questions)} questions, {len(documents)} documents from {dataset_name}")
    logging.info(f"Naive config: {naive_config}")
 
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    t_total_start = time.time()
 
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_questions)):
        fold_num = fold_idx + 1
        logging.info(f"Fold {fold_num}/{n_folds}: train={len(train_idx)} test={len(test_idx)}")
 
        train_questions = [all_questions[i] for i in train_idx]
        test_questions = [all_questions[i] for i in test_idx]
 
        fold_dir = out / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        answers_dir = fold_dir / "trial_answers"
        answers_dir.mkdir(parents=True, exist_ok=True)
 
        with open(fold_dir / "fold_split.json", "w") as f:
            json.dump({"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}, f, indent=2)
 
        t_start = time.time()
        index, chunks = build_index(
            documents, embed_model,
            naive_config["chunk_size"], naive_config["chunk_overlap"]
        )
 
        logging.info(f"Fold {fold_num}: running on train questions")
        train_records, train_failed, t_search_tr, t_gen_tr = run_questions(
            train_questions, index, chunks, embed_model, naive_config["top_k"], dataset_name
        )
 
        with open(answers_dir / "trial_1_answers.json", "w") as f:
            json.dump({
                "trial_id": 1,
                "config": naive_config,
                "num_failed": len(train_failed),
                "failed_indices": train_failed,
                "qa_pairs": train_records,
            }, f, indent=2, default=str)
 
        logging.info(f"Fold {fold_num}: running on test questions")
        test_records, test_failed, t_search_te, t_gen_te = run_questions(
            test_questions, index, chunks, embed_model, naive_config["top_k"], dataset_name
        )
 
        total_time = time.time() - t_start
        t_search = t_search_tr + t_search_te
        t_gen = t_gen_tr + t_gen_te
 
        with open(fold_dir / "test_answers.json", "w") as f:
            json.dump({
                "fold": fold_num,
                "config": naive_config,
                "num_failed": len(test_failed),
                "failed_indices": test_failed,
                "qa_pairs": test_records,
            }, f, indent=2, default=str)
 
        with open(fold_dir / "naive_results.json", "w") as f:
            json.dump({
                "fold": fold_num,
                "config": naive_config,
                "time_s": round(total_time, 2),
                "search_s": round(t_search, 2),
                "gen_s": round(t_gen, 2),
                "num_chunks": len(chunks),
                "num_failed_train": len(train_failed),
                "num_failed_test": len(test_failed),
            }, f, indent=2, default=str)
 
        logging.info(f"Fold {fold_num}: done in {total_time:.1f}s, "
                     f"train_failed={len(train_failed)} test_failed={len(test_failed)}")
 
        fold_results.append({
            "fold": fold_num,
            "config": naive_config,
            "time_s": round(total_time, 2),
        })
 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    total_time = time.time() - t_total_start
 
    summary = {
        "dataset": dataset_name,
        "n_folds": n_folds,
        "config": naive_config,
        "fold_results": fold_results,
        "total_time_s": round(total_time, 2),
    }
 
    with open(out / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
 
    logging.info(f"Done in {total_time:.0f}s")
    return summary
 
 
if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiEval")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="naive_rag_results")
    args = parser.parse_args()
 
    run(args.dataset, args.n_folds, args.output_dir)
