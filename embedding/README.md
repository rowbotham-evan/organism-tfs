# Embedding generation

Run this script from the repository root after building the RegulonDB and
UniProt datasets.

| File | What it does |
| --- | --- |
| `encode_all_tfs.py` | Loads the selected model from `models/`, embeds every amino acid in every TF sequence, removes special-token vectors, validates the model-specific width, and saves the tensors in the matching subdirectory here. |
| `extract_first_tfs.py` | Exports the first raw E. coli and PAO1 TF embeddings as model-specific residue-level CSV files without PCA. |
| `analyze_600m_clusters.py` | For each organism cluster of mean-pooled TF vectors: the mean vector (saved as CSV), the RMS in the cluster, and the pairwise cosine similarity between mean vectors. |
| `esmc-600M/` | Contains embeddings generated with ESMC-600M. |

Choose the dataset with flags:

```bash
uv run python embedding/encode_all_tfs.py --model 600m
uv run python embedding/encode_all_tfs.py --dataset pao1 --model 600m
uv run python embedding/encode_all_tfs.py --dataset nrc1 --model 600m
uv run python embedding/extract_first_tfs.py
uv run python embedding/analyze_600m_clusters.py
```

ESMC-600M produces 1,152 values per residue and writes
`embedding/esmc-600M/esmc-600M_all_raw_embeddings.pt`.

*P. aeruginosa* PAO1 outputs use the same model directories and include
`pseudomonas_aeruginosa_pao1` in their filenames.

*H. salinarum* NRC-1 outputs follow the same convention with
`halobacterium_salinarum_nrc1` in their filenames.
