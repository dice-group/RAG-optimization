import logging
import re
import requests
import load_config as config

JUDGE_SYSTEM_PROMPT = config.answer_correctness_prompt
JUDGE_TEMPLATE = config.answer_correctness_template

def parse_yes_no(text):
    match = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).lower() == "yes" else 0.0
    return None


def judge_answer(question, golden_answer, generated_answer):
    prompt = JUDGE_TEMPLATE.format(
        system_prompt=JUDGE_SYSTEM_PROMPT,
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
    score = parse_yes_no(raw)
    return score, raw


def evaluate_correctness_judge(questions, predictions, goldens):
    all_scores = []
    all_details = []

    for i in range(len(predictions)):
        if predictions[i] == "":
            all_scores.append(0.0)
            all_details.append({"score": 0.0, "raw": ""})
            continue

        score, raw = judge_answer(questions[i]["question"], goldens[i], predictions[i])
        if score is None:
            score = 0.0
            logging.warning(f"Q{i}: failed to parse yes/no, defaulting to 0")
        all_scores.append(score)
        all_details.append({"score": score, "raw": raw})

    return all_scores, all_details