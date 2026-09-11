# ESMC models

This directory contains the local ESM Cambrian checkpoints. Model weights are
excluded from Git because they are large and reproducible; derived embeddings
live under `embedding/`.

| Path | Purpose |
| --- | --- |
| `esmc-600M/` | Hugging Face-native `biohub/ESMC-600M-hf` checkpoint; produces a 1,152-value vector per amino acid. |

Each model directory contains:

| File | Why it is required |
| --- | --- |
| `config.json` | Defines the ESMC architecture. |
| `model.safetensors` | Contains the learned model weights. |
| `tokenizer.json` | Converts amino-acid sequences to token IDs. |
| `tokenizer_config.json` | Configures tokenization and special tokens. |

Generate model-specific embeddings from the repository root:

```bash
uv run python embedding/encode_all_tfs.py --model 600m
```
