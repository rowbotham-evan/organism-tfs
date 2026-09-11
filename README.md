# Transcription-factor datasets and ESMC embeddings

This project builds a catalog of *E. coli* K-12 transcription factors, fetches
their canonical amino-acid sequences, and embeds every residue with a selected
ESM Cambrian model. It also includes a UniProt-derived transcription-factor
catalog for *Pseudomonas aeruginosa* PAO1.

## Data pipeline

Run these commands from the repository root:

```bash
uv run python data/ecoli/get_regulon_tf_dataset.py
uv run python data/ecoli/get_aa_seq_from_uniprot.py
uv run python data/pao1/get_mist_signal_genes.py
uv run python data/pao1/get_pseudomonas_aeruginosa_pao1_tf_dataset.py
uv run python data/nrc1/get_network_portal_tfs.py
uv run python data/nrc1/get_halobacterium_salinarum_nrc1_tf_dataset.py
uv run python data/network_portal/get_halobacterium_salinarum_nrc1_tfs.py
uv run python embedding/encode_all_tfs.py --model 300m
# Or use ESMC-600M:
uv run python embedding/encode_all_tfs.py --model 600m
uv run python embedding/encode_all_tfs.py --dataset pao1 --model 300m
uv run python embedding/encode_all_tfs.py --dataset pao1 --model 600m
uv run python embedding/encode_all_tfs.py --dataset nrc1 --model 600m
uv run python visualization/plot_tf_sequence_length_distribution.py
uv run python visualization/plot_raw_embedding_pca.py
uv run python visualization/plot_pooled_embedding_pca.py
uv run python visualization/plot_pooled_embedding_pca_components.py
uv run python visualization/plot_pooled_embedding_tsne.py
uv run python visualization/plot_pooled_embedding_umap.py
uv run python visualization/plot_tf_nearest_neighbour_distances.py
uv run python embedding/analyze_600m_clusters.py
```

1. `get_regulon_tf_dataset.py` makes one GraphQL request for RegulonDB's
   `TFSet` and `GeneProductAllIdentifiersSet`. It saves only the raw `TFSet`;
   the identifier table is used in memory to join UniProt accessions by
   b-number. The cleaned output has one row per unique TF protein.
2. `get_aa_seq_from_uniprot.py` looks up each accession directly in UniProtKB
   and appends the canonical complete sequence.
3. `pao1/get_mist_signal_genes.py` collects the P. aeruginosa PAO1 genes on
   MiST4's `kind=output`, `function=DNA binding` page (its transcription factor
   category, including sigma factors), with each gene's locus tag and the
   genome's organism ID. `pao1/get_pseudomonas_aeruginosa_pao1_tf_dataset.py`
   then queries UniProtKB for `(organism_id) AND (gene:<locus tag>) AND
   (go:0003700)`, re-checks the organism, exact locus tag and TF GO term
   (GO:0003700 or one of its child terms from QuickGO) on every returned entry,
   and saves the sequences. MiST4 has no GO annotations; GO:0003700 is the
   criterion the pipeline adds.
3b. `nrc1/get_network_portal_tfs.py` downloads the Baliga lab catalog from the
   ISB Network Portal, then `nrc1/get_halobacterium_salinarum_nrc1_tf_dataset.py`
   selects NRC-1 proteins carrying GO:0003700 or a Pfam helix-turn-helix family,
   unions them with that catalog, and drops entries whose UniProt name is
   affirmatively non-regulatory.
3c. `data/network_portal/get_halobacterium_salinarum_nrc1_tfs.py` downloads the
   Baliga Lab Network Portal's complete NRC-1 TF table and refuses to write the
   output unless the reported count, parsed row count, unique locus-tag count,
   and TF flags agree.
4. `embedding/encode_all_tfs.py --dataset {ecoli,pao1,nrc1} --model {300m,600m}`
   produces one model-width ESMC vector per residue and writes a model- and
   dataset-named embedding file.
5. `plot_raw_embedding_pca.py` runs PCA directly on all residue-level
   ESMC-600M embeddings, with E. coli, P. aeruginosa PAO1, and
   H. salinarum NRC-1 colored separately.
6. `plot_tf_sequence_length_distribution.py` validates every ESMC-600M
   embedding tensor and plots all three organisms' protein lengths in
   50-amino-acid bins as one stacked panel with a shared x-axis.
7. `plot_pooled_embedding_pca.py` mean-pools each TF into one vector and runs
   PCA over proteins rather than residues.
   `plot_pooled_embedding_pca_components.py` computes the first eight
   components of that PCA and plots each new one against the previous
   (PC2-PC3 through PC7-PC8), reporting how much of each component's variance
   organism and domain of life explain.
   `plot_pooled_embedding_tsne.py` runs t-SNE on the same pooled vectors
   (50 PCs, perplexity 30, fixed seed) and reports how often each TF's nearest
   neighbours in the original embedding space come from its own organism.
   `plot_pooled_embedding_umap.py` runs UMAP on the same pooled vectors
   (50 PCs, 15 neighbours, min_dist 0.1, fixed seed) and reports how well the
   2-d layout preserves each TF's original nearest neighbours.
8. `plot_tf_nearest_neighbour_distances.py` gives each TF's Euclidean distance
   to the closest other TF in the same organism, as one stacked histogram panel.
9. `analyze_600m_clusters.py` reports each organism cluster's mean vector, its
   norm and RMS, plus the cosine similarity and Euclidean distance between
   every pair of mean vectors, in both pooled-protein and raw-residue space.
   Writes `results/ESMC-600M_cluster_statistics.csv` and
   `results/ESMC-600M_cluster_pairwise.csv`. ESMC-600M only.

The primary tables are:

- `data/ecoli/data/regulon_tf_dataset_raw.tsv`: unmodified RegulonDB TFSet.
- `data/ecoli/data/regulon_tf_dataset_clean.csv`: one row per TF protein, with
  b-number, TF entity metadata, RegulonDB version, and UniProt accession.
- `data/ecoli/data/regulon_tf_dataset_with_aa_sequences.csv`: the clean catalog
  plus canonical UniProt sequences.
- `data/pao1/data/pseudomonas_aeruginosa_pao1_mist_dna_binding_genes.csv`: MiST4
  signal genes for GCF_000006765.1 with `kind=output` and `function=DNA binding`.
- `data/pao1/data/pseudomonas_aeruginosa_pao1_tf_dataset.csv`: MiST4 DNA-binding
  genes whose UniProtKB entry matches the MiST organism ID, the locus tag, and
  GO:0003700 or a child term, with sequences.
- `data/nrc1/data/halobacterium_salinarum_nrc1_tf_dataset.csv`: complete
  NRC-1 proteins matching
  `(organism_id:64091) AND (go:0003700) AND (fragment:false)`.
- `data/network_portal/data/halobacterium_salinarum_nrc1_network_portal_tfs.csv`:
  all 125 NRC-1 genes marked as TFs by the Baliga Lab Network Portal, retaining
  its legacy VNG tags, NCBI Gene IDs, coordinates, and descriptions.

`data/` is organised one directory per organism (`ecoli/`, `pao1/`, `nrc1/`, `hvolc/`),
each holding its download scripts and a `data/` subdirectory of outputs.

These tables are model-independent, so their names do not include `300m` or
`600m`. Only checkpoints and derived embeddings are model-specific.

See [`models/README.md`](models/README.md) for the local checkpoints and
[`embedding/README.md`](embedding/README.md) for embedding generation and
model-specific outputs.
Plots are stored under `results/`. Visualization is ESMC-600M only for now;
`encode_all_tfs.py` still accepts `--model 300m` for generating embeddings.

## Root files

| File | Purpose |
| --- | --- |
| `.gitignore` | Excludes the virtual environment, caches, downloaded ESMC weights, generated embeddings, and analysis results from Git. |
| `.python-version` | Pins the Python version used by `uv`. |
| `README.md` | Documents the project structure and shortest end-to-end workflow. |
| `pyproject.toml` | Declares project metadata, supported Python, and direct dependencies. |
| `uv.lock` | Locks the complete dependency graph for reproducible installs. |
