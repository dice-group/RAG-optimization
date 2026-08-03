# DO WE NEED AN EXPENSIVE JUDGE FOR RAG OPTIMIZATION

This repository contains the code for implementation and the evaluation of this work. 

## Repository structure
```
|-- config.json
|-- load_config.py
|-- logging_setup.py
|-- generator.py
|-- indexer.py
|-- retriever.py
|-- pipeline.py
|-- naive_rag.py
|-- run_optimization.py
|-- ragas_evaluation.py
|-- generate_negatives.py
|-- prepare_hotpotqa.py
|-- prepare_bioasq.py
|-- requirements.txt
|-- optimization_methods/
|   |-- bertscore.py
|   |-- answer_correctness.py
|   |-- persona.py
|   |-- contrastive.py
|   |-- token_recall.py
|-- results/
|   |-- per_fold_results.csv
|   |-- optimization_trials.csv
|   |-- friedman_results.csv
|   |-- nemenyi_pairwise_comparisons_all_methods.csv
```

## FILE DESCRIPTIONS

### Configuration Files
- config.json: Contains the default RAG parameters and prompt templates used in the pipeline. Additionally, it also includes the embedding model, retrieval depth, and generation limits.
- load_config.py: Loads the values defined in the configuration file.
- logging_setup.py: Configures logging for the executable scripts. The logs are written to logs/pipeline.log

### Dataset Preparation
- prepare_hotpotqa.py: Selects a subset of HotpotQA questions based on question type and difficulty, collects their document contexts, and saves the questions and documents as Parquet files.
- prepare_bioasq.py: Downloads the RAGMiniBioASQ dataset, randomly selects the required number of questions, collects their relevant passages, and saves the questions and passages as Parquet files.
- generate_negatives.py: Generates the plausible incorrect answers required by the Contrastive optimization objective.
  
### RAG Pipeline
- generator.py: Constructs the generation prompt based on the input question and the data, sends the request to the language model, and returns the generated answer.
- indexer.py: Loads the documents for the dataset, divides the documents into chunks, encodes them into a sentence-transformer model, and builds a FAISS index.
- retriever.py: Loads the FAISS index, converts a question into an embedding, and returns the most similar document chunks.
- pipeline.py: Runs the complete RAG pipeline: from indexing to retrieval and generation.

### Baseline and Optimization
- naive_rag.py: Runs the RAG pipeline using the fixed settings from `config.json`. This is a non-optimized reference.
- run_optimization.py: Uses SMAC to optimize the RAG pipeline and search for better values of hyperparameters (chunk size, chunk overlap, and top-k). It evaluates on different optimization objectives and test the best configuration on the test set.

### Optimization Objectives
- answer_correctness.py: Uses a correctness judge to compare the generated answer to the corresponding reference answer.
- persona.py: Uses three persona prompts to evaluate the generated answer. 
- bertscore.py: Computes BERTScore F1 between the generated answer and the reference answer. 
- contrastive.py: A custom score that evaluates the generated answer relative to both the reference answer and a set of plausible incorrect answers.
- token_recall.py: It measures the proportion of reference answer tokens present in the generated answer.

### Final Evaluation
- ragas_evaluation.py: Loads the saved test answers and evaluates them on RAGAS Faithfulness, Answer Correctness, and Semantic Similarity.

## EXPERIMENTAL SETUP

### Requirements
This code base was developed with `Python 3.11.2`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Language Models

| Role | Model | Served with |
|---|---|---|
| Generator | [`gemma3:27b-it-qat`](https://ollama.com/library/gemma3:27b-it-qat) | Ollama |
| Generator | [`gpt-oss:20b`](https://ollama.com/library/gpt-oss:20b) | Ollama |
| Generator | [`mistral-nemo:12b`](https://ollama.com/library/mistral-nemo:12b) | Ollama |
| RAGAS judge | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)| llama.cpp |

The generator models are also used as judges for their corresponding answer correctness and persona optimization objectives. 

### Environment Variables
Create a `.env` file in the repository root. 

```bash
LLM_BASE_URL=http://localhost:11435
LLM_NAME=gpt-oss:20b 
JUDGE_BASE_URL=http://localhost:8082
JUDGE_NAME=gemma4
```
`LLM_NAME` must be set to `gemma3:27b-it-qat`, `gpt-oss:20b`, or `mistral-nemo:12b`.

### Default RAG Configuration

| Parameter | Value |
|---|---:|
| Chunk size | 512|
| Chunk overlap | 50|
| `top_k` | 5 |
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| FAISS index | `IndexFlatIP` |
| Temperature | 0 |

### Optimization Settings

The optimization is done using the following settings:

| Setting | Value |
|---|---:|
| Cross-validation folds | 5 |
| Trials per fold | 25 |
| Random seed | 42 |
| Initial configurations | 10 |
| Data shuffling | Enabled |
| Optimizer | SMAC |

### Prompts
The exact prompts for answer generation, the Correctness Judge, and the three Persona Judges are stored in `config.json` under the keys `prompt`, `answer_correctness_judge`, and `persona_judge`.

### Dataset characteristics

| Dataset | Characteristics | Original QA pairs | QA pairs used | Maximum generated tokens |
|---|---|---:|---:|---:|
| WikiEval | Wikipedia-based factual question answering | 50 | 50 | 1024 |
| HotpotQA | Multi-hop Wikipedia question answering | 113,000 | 150 | 512 |
| RAGMiniBioasq | Biomedical question answering | 4,720 | 150 | 1024 |

### Dataset preparation

#### WikiEval

WikiEval is used directly from the [Hugging Face dataset](https://huggingface.co/datasets/vibrantlabsai/WikiEval) and does not require additional preparation. Place the downloaded Parquet file in `dataset/WikiEval/data/`.

#### HotpotQA

Run the following command to sample 150 questions by question type and difficulty level and prepare the corresponding document corpus:

```bash
python prepare_hotpotqa.py \
  --input <path-to-hotpotqa-json> \
  --output_dir dataset/HotpotQA/data \
  --n 150 \
  --seed 42
```
#### RAGMiniBioasq

Run the following command to sample 150 questions and collect their relevant passages:

```bash
python prepare_bioasq.py \
  --output_dir dataset/RAGMiniBioasq/data \
  --n 150 \
  --seed 42
```

### Running the Naive baseline

Set `LLM_NAME` in the `.env` file to the generator model and run:

```bash
python naive_rag.py \
  --dataset <dataset-name> \
  --n_folds 5 \
  --output_dir <output-directory>
```

### Running the optimization

Set `LLM_NAME` in the `.env` file to the generator model being evaluated. Run the optimization using:

```bash
python run_optimization.py \
  --method <optimization-objective> \
  --dataset <dataset-name> \
  --n_trials 25 \
  --n_folds 5 \
  --output_dir <output-directory>
```
Generate the plausible incorrect answers required by the Contrastive objective:

```bash
python generate_negatives.py \
  --dataset WikiEval \
  --output negatives/WikiEval/negatives.json

python run_optimization.py \
  --method contrastive \
  --dataset WikiEval \
  --n_trials 25 \
  --n_folds 5 \
  --negatives <path-to-negative-answers> \
  --output_dir optimization_results/contrastive
```

### Final RAGAS evaluation

Ensure that the judge model server is running and that `JUDGE_BASE_URL` and `JUDGE_NAME` are defined in the `.env` file. Then run:
```bash
python ragas_evaluation.py \
  --method <method> \
  --llm <generator> \
  --dataset <dataset-name> \
  --output_dir ragas_eval_results
```
The accepted method labels are `naive`, `bertscore`, `contrastive`, `persona`, `token_recall`, and `ans_corr`, where `ans_corr` refers to the Correctness Judge. The accepted generator labels are `gemma3`, `gpt`, and `mistral`.

## Detailed experiment results
```
results/
|-- per_fold_results.csv
|-- optimization_trials.csv
|-- friedman_results.csv
|-- nemenyi_pairwise_comparisons_all_methods.csv
```
The `results/` directory contains the detailed results underlying the aggregate findings reported in the paper:

- **`per_fold_results.csv`**: RAGAS Faithfulness, Answer Correctness, and Semantic Similarity scores for every generator, dataset, cross-validation fold, and evaluated method.
- **`optimization_trials.csv`**: The hyperparameter configuration, objective score, runtime, and number of indexed chunks for each completed optimization trial.
- **`friedman_results.csv`**: The Friedman test statistic, p-value, and significance decision for each evaluation metric.
- **`nemenyi_pairwise_comparisons_all_methods.csv`**: The complete pairwise Nemenyi comparisons among all evaluated methods for each evaluation metric.