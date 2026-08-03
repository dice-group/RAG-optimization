import json
import os
from dotenv import load_dotenv

load_dotenv()

config_path = os.environ.get("CONFIG_PATH", "config.json")

with open(config_path) as f:
    config = json.load(f)

chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]
embedding_model = config["embedding_model"]
embedding_dimension = config["vector_size"][config["embedding_model"]]
top_k = config["top_k"]
num_predict = config["num_predict"]
prompt_template = config["prompt"]
persona_prompts = config["persona_judge"]["personas"]
persona_judge_template = config["persona_judge"]["template"]
answer_correctness_prompt = config["answer_correctness_judge"]["system_prompt"]
answer_correctness_template = config["answer_correctness_judge"]["template"]


llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11435")
llm_name = os.environ.get("LLM_NAME", "gpt-oss:20b")