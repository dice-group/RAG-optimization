import os
import logging

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
        force=True,
    )

    logging.captureWarnings(True)

    for name in ("sentence_transformers","transformers", "safetensors", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.info("Pipeline started running")