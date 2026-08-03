import torch 
from bert_score import BERTScorer

bert_device = "cuda" if torch.cuda.is_available() else "cpu"
bert_batch_size = 16
scorer = BERTScorer(lang="en", device=bert_device, batch_size=bert_batch_size)

def evaluate_bertscore(predictions, goldens):
    scores = [0.0] * len(predictions)

    valid_indices = [i for i, prediction in enumerate(predictions) if prediction.strip()]

    if not valid_indices:
        return scores

    valid_predictions = [predictions[i] for i in valid_indices]
    valid_goldens = [goldens[i] for i in valid_indices]

    _, _, F1 = scorer.score(valid_predictions, valid_goldens)

    for index, value in zip(valid_indices, F1):
        scores[index] = max(0.0, float(value.item()))
    return scores