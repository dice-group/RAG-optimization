import logging
import re
import numpy as np
import requests
import load_config as config

PERSONAS = config.persona_prompts
JUDGE_TEMPLATE = config.persona_judge_template

def parse_rating(text):
    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if match:
        return float(match.group(1))
    return None


def judge_answer(question, golden_answer, generated_answer, persona_name):
    prompt = JUDGE_TEMPLATE.format(
        system_prompt=PERSONAS[persona_name],
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    body = {
        "model": config.llm_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    if "gpt-oss" in config.llm_name:
        body["think"] = "low"
    response = requests.post(f"{config.llm_base_url}/api/chat", json=body, timeout=300)
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    rating = parse_rating(raw)
    return rating, raw


def evaluate_persona_judges(questions, predictions, goldens):
    persona_names = list(PERSONAS.keys())
    all_scores = []
    all_details = []

    for i in range(len(predictions)):
        if predictions[i] == "":
            all_scores.append(0.0)
            all_details.append({p: {"score": 0.0, "raw": ""} for p in persona_names})
            continue

        question_scores = {}
        for persona in persona_names:
            rating, raw = judge_answer(questions[i]["question"], goldens[i], predictions[i], persona)
            if rating is None:
                rating = 1.0
                logging.warning(f"Q{i} persona={persona}: failed to parse rating, defaulting to 1")
            score = min(rating / 10.0, 1.0)
            question_scores[persona] = {"score": score, "raw": raw}

        avg = float(np.mean([question_scores[p]["score"] for p in persona_names]))
        all_scores.append(avg)
        all_details.append(question_scores)

    return all_scores, all_details