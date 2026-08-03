import os
import json
import numpy as np
import faiss
import argparse
import logging
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import load_config as config

def load_wikieval(dataset_path):
    table = pq.read_table(dataset_path)
    documents = []
    for i in range(table.num_rows):
        source = table.column("source")[i].as_py()
        for passage in table.column("context_v1")[i].as_py():
            documents.append({"content": passage, "source": source})
    return documents


def load_hotpotqa_ragbioasq(dataset_path):
    doc_path = os.path.join(os.path.dirname(dataset_path), "documents.parquet")
    table = pq.read_table(doc_path)
    documents = []
    for i in range(table.num_rows):
        documents.append({
            "content": table.column("content")[i].as_py(),
            "source": table.column("source")[i].as_py(),
        })
    return documents


DATASET_LOADERS = {
    "WikiEval": load_wikieval,
    "HotpotQA": load_hotpotqa_ragbioasq,
    "RAGMiniBioasq": load_hotpotqa_ragbioasq,
}


def load_documents(dataset_name, dataset_path):
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"No loader for '{dataset_name}'. Available: {list(DATASET_LOADERS.keys())}")
    return loader(dataset_path)


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = []
    for doc in documents:
        for split in splitter.split_text(doc["content"]):
            chunks.append({"text": split, "source": doc["source"]})
    return chunks


def build_index(chunks, index_dir):
    model = SentenceTransformer(config.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    index = faiss.IndexFlatIP(config.embedding_dimension)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)


def run(dataset_name, dataset_path, index_dir):
    documents = load_documents(dataset_name, dataset_path)
    chunks = chunk_documents(documents)
    build_index(chunks, index_dir)
    logging.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents into {index_dir}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--index_dir", default="faiss_index")
    args = parser.parse_args()
    run(args.dataset, args.dataset_path, args.index_dir)