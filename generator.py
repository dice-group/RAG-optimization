import requests
import load_config as config


def generate(question, chunks, dataset):
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = config.prompt_template.format(context=context, question=question)
    body = {
        "model": config.llm_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": config.num_predict[dataset]},
    }
    if "gpt-oss" in config.llm_name:
        body["think"] = "low"
    response = requests.post(f"{config.llm_base_url}/api/chat", json=body, timeout=600)
    response.raise_for_status()
    return response.json()["message"]["content"]
 