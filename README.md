# Transcription-factor datasets and ESMC embeddings

This project builds a catalog of *E. coli* K-12 transcription factors, fetches
their canonical amino-acid sequences, and embeds every residue with a selected
ESM Cambrian model. It also includes a UniProt-derived transcription-factor
catalog for *Pseudomonas aeruginosa* PAO1.

## Data pipeline

Run these commands from the repository root:

```bash
uv run python data/Escherichia_coli_K-12/get_regulon_tf_dataset.py
uv run python data/Escherichia_coli_K-12/get_aa_seq_from_uniprot.py
uv run python data/Pseudomonas_aeruginosa_PAO1/get_mist_signal_genes.py
uv run python data/Pseudomonas_aeruginosa_PAO1/get_pseudomonas_aeruginosa_pao1_tf_dataset.py
uv run python data/Halobacterium_salinarum_NRC-1/get_mist_signal_genes.py
uv run python data/Halobacterium_salinarum_NRC-1/get_halobacterium_salinarum_nrc1_tf_dataset.py
uv run python embedding/encode_all_tfs.py --model 600m
uv run python embedding/encode_all_tfs.py --dataset pao1 --model 600m
uv run python embedding/encode_all_tfs.py --dataset nrc1 --model 600m
uv run python visualization/plot_tf_sequence_length_distribution.py
uv run python visualization/plot_raw_embedding_pca.py
uv run python visualization/plot_pooled_embedding_pca.py
uv run python visualization/plot_pooled_embedding_pca_components.py
uv run python embedding/extract_layer_pooled_embeddings.py
uv run python visualization/plot_pooled_embedding_pca_by_layer.py
uv run python visualization/plot_pooled_embedding_tsne.py
uv run python visualization/plot_pooled_embedding_tsne_perplexity.py
uv run python visualization/plot_pooled_embedding_umap.py
uv run python visualization/plot_pooled_embedding_umap_grid.py
uv run python visualization/plot_tf_distance_panels.py
uv run python embedding/analyze_600m_clusters.py
```

1. `get_regulon_tf_dataset.py` makes one GraphQL request for RegulonDB's
   `TFSet` and `GeneProductAllIdentifiersSet`. It saves only the raw `TFSet`;
   the identifier table is used in memory to join UniProt accessions by
   b-number. The cleaned output has one row per unique TF protein.
2. `get_aa_seq_from_uniprot.py` looks up each accession directly in UniProtKB
   and appends the canonical complete sequence.
3. `Pseudomonas_aeruginosa_PAO1/get_mist_signal_genes.py` collects the P. aeruginosa PAO1 genes on
   MiST4's `kind=output`, `function=DNA binding` page (its transcription factor
   category, including sigma factors), with each gene's locus tag and the
   genome's organism ID. `Pseudomonas_aeruginosa_PAO1/get_pseudomonas_aeruginosa_pao1_tf_dataset.py`
   then queries UniProtKB for `(organism_id) AND (gene:<locus tag>) AND
   (go:0003700)`, re-checks the organism, exact locus tag and TF GO term
   (GO:0003700 or one of its child terms from QuickGO) on every returned entry,
   and saves the sequences. MiST4 has no GO annotations; GO:0003700 is the
   criterion the pipeline adds.
3b. `Halobacterium_salinarum_NRC-1/get_mist_signal_genes.py` collects the H. salinarum NRC-1 genes on
   MiST4's `kind=output`, `function=DNA binding` page for genome GCF_000006805.1.
   `Halobacterium_salinarum_NRC-1/get_halobacterium_salinarum_nrc1_tf_dataset.py` takes the union of those
   genes and UniProtKB `(organism_id:64091) AND (go:0003700) AND (fragment:false)`.
   MiST genes are linked to UniProt by locus tag (MiST's old tag VNG6478H written
   as VNG_6478H) or RefSeq protein accession; linked genes use UniProt sequences,
   unlinked genes use the RefSeq sequence served by MiST4, and identical plasmid
   copies collapse to one row.
4. `embedding/encode_all_tfs.py --dataset {ecoli,pao1,nrc1,hvolc} --model 600m`
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
   components of that PCA and draws PC2-PC3 through PC7-PC8 as panels of one
   figure, reporting how much of each component's variance organism and domain
   of life explain.
   `extract_layer_pooled_embeddings.py` reruns ESMC-600M with all hidden states
   and keeps each TF's mean-pooled vector at layers 2, 4, 6, ... 36 (36 is the
   final LayerNorm output used everywhere else).
   `plot_pooled_embedding_pca_by_layer.py` fits a separate PCA per layer and
   draws the 18 of them as a grid in one figure.
   `plot_pooled_embedding_tsne.py` runs t-SNE on the same pooled vectors
   (50 PCs, perplexity 30, fixed seed) and reports how often each TF's nearest
   neighbours in the original embedding space come from its own organism.
   `plot_pooled_embedding_tsne_perplexity.py` repeats that t-SNE at perplexity
   5, 10, 15, ... 50 as a 2x5 grid in one figure.
   `plot_pooled_embedding_umap.py` runs UMAP on the same pooled vectors
   (50 PCs, 15 neighbours, min_dist 0.1, fixed seed) and reports how well the
   2-d layout preserves each TF's original nearest neighbours.
   `plot_pooled_embedding_umap_grid.py` repeats that UMAP with n_neighbors
   5, 10, ... 50 as columns and min_dist 0.0, 0.1, 0.5 as rows in one figure,
   with each panel's trustworthiness.
8. `plot_tf_distance_panels.py` writes one figure per organism
   (`results/ESMC-600M_{ecoli,pao1,nrc1,hvolc}_tf_distances.png`). Each stacks
   four histograms of every TF's Euclidean distance to its nearest TF in each
   organism: its own organism first (nearest other self TF), then the other
   three, on a shared distance axis.
9. `analyze_600m_clusters.py` reports each organism cluster's mean vector, its
   norm and RMS, plus the cosine similarity and Euclidean distance between
   every pair of mean vectors, in both pooled-protein and raw-residue space.
   Writes `results/ESMC-600M_cluster_statistics.csv` and
   `results/ESMC-600M_cluster_pairwise.csv`. ESMC-600M only.

The primary tables are:

- `data/Escherichia_coli_K-12/data/regulon_tf_dataset_raw.tsv`: unmodified RegulonDB TFSet.
- `data/Escherichia_coli_K-12/data/regulon_tf_dataset_clean.csv`: one row per TF protein, with
  b-number, TF entity metadata, RegulonDB version, and UniProt accession.
- `data/Escherichia_coli_K-12/data/regulon_tf_dataset_with_aa_sequences.csv`: the clean catalog
  plus canonical UniProt sequences.
- `data/Pseudomonas_aeruginosa_PAO1/data/pseudomonas_aeruginosa_pao1_mist_dna_binding_genes.csv`: MiST4
  signal genes for GCF_000006765.1 with `kind=output` and `function=DNA binding`.
- `data/Pseudomonas_aeruginosa_PAO1/data/pseudomonas_aeruginosa_pao1_tf_dataset.csv`: MiST4 DNA-binding
  genes whose UniProtKB entry matches the MiST organism ID, the locus tag, and
  GO:0003700 or a child term, with sequences.
- `data/Halobacterium_salinarum_NRC-1/data/halobacterium_salinarum_nrc1_mist_dna_binding_genes.csv`: MiST4
  signal genes for GCF_000006805.1 with `kind=output` and `function=DNA binding`,
  including old locus tags and RefSeq protein accessions.
- `data/Halobacterium_salinarum_NRC-1/data/halobacterium_salinarum_nrc1_tf_dataset.csv`: union of those
  genes and UniProt GO:0003700 TFs, one row per distinct protein, with source
  flags (`in_mist`, `in_uniprot_go`) and how each MiST gene was linked.

`data/` is organised one directory per organism (`Escherichia_coli_K-12/`, `Pseudomonas_aeruginosa_PAO1/`, `Halobacterium_salinarum_NRC-1/`, `Haloferax_volcanii_DS2/`),
each holding its download scripts and a `data/` subdirectory of outputs.

These tables are model-independent, so their names do not include the model
name. Only checkpoints and derived embeddings are model-specific.

See [`models/README.md`](models/README.md) for the local checkpoints and
[`embedding/README.md`](embedding/README.md) for embedding generation and
model-specific outputs.
Plots are stored under `results/`; PCA, t-SNE and UMAP figures are in
`results/dimensionality_reduction/`. The project uses ESMC-600M only.

## Root files

| File | Purpose |
| --- | --- |
| `.gitignore` | Excludes the virtual environment, caches, downloaded ESMC weights, generated embeddings, and analysis results from Git. |
| `.python-version` | Pins the Python version used by `uv`. |
| `README.md` | Documents the project structure and shortest end-to-end workflow. |
| `pyproject.toml` | Declares project metadata, supported Python, and direct dependencies. |
| `uv.lock` | Locks the complete dependency graph for reproducible installs. |
