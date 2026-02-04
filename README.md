# ICRL: In-Context Reinforcement Learning for Tool Use in Large Language Models

A framework for training Large Language Models to use external tools (e.g., search engines) through in-context learning combined with reinforcement learning.

## Installation

### Install Environment

```bash
git clone https://github.com/applese233/ICRL.git
cd ICRL

# Environment with Python 3.9
conda env create -f environment.yml
conda activate icrl

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

pip install -e .

pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Prepare Data

```bash
python scripts/data_process/nq_search_fewshot.py \
    --output_dir data/nq_search_fewshot \
    --num_examples 3
```

### 2. Start Search Server

The training requires a web search server. We use SerpAPI or Serper.dev as the search backend.

First, get your API key from:
- [SerpAPI](https://serpapi.com) (default)
- [Serper.dev](https://serper.dev)

Then start the server:

```bash
# Using SerpAPI (default)
SERPAPI_KEY=your_api_key bash scripts/search/run_search_server.sh

# Or using Serper.dev
PROVIDER=serper SERPAPI_KEY=your_api_key bash scripts/search/run_search_server.sh
```

The server will listen on `http://127.0.0.1:8000/retrieve`.

### 3. Train with GRPO

You can train the model with a single stage:
```bash
bash train_grpo_fewshot.sh
```
Or train the whole method:
```bash
bash train_curriculum.sh
```

### 4. Inference

Test the model with a single question:

```bash
# Ask a question directly
python infer.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --question "Who won the 2022 FIFA World Cup?"

# Interactive mode (enter questions one by one)
python infer.py --model_path your_trained_model

# Use few-shot prompts
python infer.py \
    --model_path your_model \
    --use_fewshot \
    --fewshot_path example/fewshot_examples.txt \
    --question "Your question here"
```

### 5. Evaluate

Evaluate on benchmark datasets:

```bash
bash eval_batch_vllm.sh
```

## Evaluation Datasets

- TriviaQA
- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- Bamboogle

## Acknowledgements

This project builds upon:
- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - Search-augmented LLM training

## License

Apache License 2.0
