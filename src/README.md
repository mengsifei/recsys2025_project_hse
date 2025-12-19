# Recsys2025 Solutions Analysis and Experiments

Generate multiple user embeddings (ALS / LightFM / smallLLM) and **merge** them into one matrix aligned by `client_id`.

## Structure

```
src/
├── ALS/
│   └── run_als.py
├── LightFM/
│   └── run_fm.py
├── smallLLM/
│   ├── embed_clients_vllm.py
│   ├── prepare_interactions_text.py
│   └── prepare_search_queries_jsonl.py
├── merge/
│   └── merge_all_solutions.py
├── helpers/
│   └── data.py
├── EDA/
│   └── eda.ipynb
├── config/
│   ├── all_interactions_text.yaml
│   ├── buy_add_categories.yaml
│   ├── lightfm_64.yaml
│   ├── merge.yaml
│   ├── page_session.yaml
│   ├── search_queries_text.yaml
│   └── small_llm.yaml
├── pyproject.toml
└── README.md
```

## Setup

```bash
poetry install
poetry shell
export COMP_DATA=/path/to/challenge_dataset
```

Configs are loaded as `src/config/<name>.yaml` via `--config=<name>`.

## Run

### ALS

```bash
poetry run python -m ALS.run_als --config=buy_add_categories
poetry run python -m ALS.run_als --config=page_session
```

### LightFM

```bash
poetry run python -m LightFM.run_fm --config=lightfm_64
```

### smallLLM (text -> vLLM embeddings)

```bash
poetry run python -m smallLLM.prepare_interactions_text --config=all_interactions_text
poetry run python -m smallLLM.prepare_search_queries_jsonl --config=search_queries_text
poetry run python -m smallLLM.embed_clients_vllm --config=small_llm
```

### Merge everything

```bash
poetry run python -m merge.merge_all_solutions --config=merge
```

Outputs are saved under:

```
$COMP_DATA/embeddings/<run_name>/{client_ids.npy, embeddings.npy}
```

What merge does:
- Loads multiple embedding folders (each must contain client_ids.npy + embeddings.npy)
- Optionally drops all-zero rows (per-input)
- Applies per-input transforms (e.g., PCA, Normalizer)
- Aligns by union of users
- Optionally pads to all relevant clients
- Imputes missing values (e.g., mean) and saves final matrix