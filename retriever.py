import json
import faiss
from sentence_transformers import SentenceTransformer
import load_config as config

_model = None

def _get_model():
    global _model
    if _model is None:
        _model =  SentenceTransformer(config.embedding_model)
    return _model


def load_index(index_dir):
    index = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/chunks.json") as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query, index, chunks, top_k=None):
    if top_k is None:
        top_k = config.top_k
    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({"text": chunks[idx]["text"], "source": chunks[idx]["source"], "score": float(score)})
    return results