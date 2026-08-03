import json
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def contrastive_score(generated_emb, golden_emb, negative_embs):
    sim_xg = cosine_sim(generated_emb, golden_emb) + 1
    sim_fgs = [cosine_sim(neg, golden_emb) + 1 for neg in negative_embs]
    max_sim_fg = max(sim_fgs)

    denom = 2 - max_sim_fg
    if denom < 1e-8:
        denom = 1e-8

    ratio = (sim_xg - max_sim_fg) / denom
    return 1 / (1 + np.exp(-ratio))

def evaluate_contrastive(predictions, goldens, negatives_map, embed_model):
    all_texts = predictions + goldens
    for q_negs in negatives_map:
        all_texts.extend(q_negs)

    embeddings = embed_model.encode(
        all_texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )

    n = len(predictions)
    pred_embs = embeddings[:n]
    gold_embs = embeddings[n:2*n]

    offset = 2 * n
    scores = []
    for i in range(n):
        neg_count = len(negatives_map[i])
        neg_embs = embeddings[offset:offset + neg_count]
        offset += neg_count

        if len(neg_embs) == 0 or predictions[i] == "":
            scores.append(0.0)
            continue

        s = contrastive_score(pred_embs[i], gold_embs[i], neg_embs)
        scores.append(float(s))

    return scores

def load_negatives(negatives_path):
    with open(negatives_path) as f:
        data = json.load(f)
    return [entry["negatives"] for entry in data]