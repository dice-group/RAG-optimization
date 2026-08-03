import re
import string
from collections import Counter

def normalize_answer(text):
    if not text:
        return ""
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def get_tokens(text):
    if not text:
        return []
    return normalize_answer(text).split()


def token_recall(prediction, reference):
    reference_tokens = Counter(get_tokens(reference))
    prediction_tokens = Counter(get_tokens(prediction))
    common_tokens = reference_tokens & prediction_tokens
    num_common = sum(common_tokens.values())
    if sum(reference_tokens.values()) == 0:
        return 1.0
    if num_common == 0:
        return 0.0
    return 1.0 * num_common / sum(reference_tokens.values())


def evaluate_token_recall(predictions, goldens):
    return [token_recall(p, g) for p, g in zip(predictions, goldens)]