"""Plot a protein-level UMAP of mean-pooled ESMC-600M embeddings.

Uses the same inputs as plot_pooled_embedding_tsne.py: one mean-pooled vector
per TF, reduced to 50 principal components. Like t-SNE, UMAP is built to keep
local neighbourhoods, so distances between clusters are not directly
interpretable. The script reports trustworthiness, how well each TF's
nearest neighbours in the embedding space survive into the 2-d layout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import torch
import umap


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
PCA_COMPONENTS = 50
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 0
TRUST_NEIGHBOURS = 10
DATASETS = (
    {
        "name": "E. coli K-12",
        "suffix": "all_raw_embeddings.pt",
        "color": "#0072FF",
    },
    {
        "name": "P. aeruginosa PAO1",
        "suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "color": "#FF6B00",
    },
    {
        "name": "H. volcanii DS2",
        "suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "color": "#00A6A6",
    },
    {
        "name": "H. salinarum NRC-1",
        "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "color": "#7B3FF2",
    },
)


def pool_dataset(dataset):
    """One mean-pooled vector per protein."""
    records = torch.load(
        PROJECT_DIR / "embedding" / MODEL["slug"]
        / f"{MODEL['slug']}_{dataset['suffix']}",
        map_location="cpu",
        weights_only=False,
    )

    pooled = []
    for record in records:
        raw = record["embedding"].float()
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, got {tuple(raw.shape)}"
            )
        pooled.append(raw.mean(dim=0))
    return torch.stack(pooled)


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    matrix = torch.cat(blocks).numpy().astype(np.float64)
    labels = np.concatenate([
        np.full(block.shape[0], index) for index, block in enumerate(blocks)
    ])

    reduced = PCA(
        n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
    ).fit_transform(matrix)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    coordinates = reducer.fit_transform(reduced)
    trust = trustworthiness(matrix, coordinates, n_neighbors=TRUST_NEIGHBOURS)

    fig, ax = plt.subplots(figsize=(9, 8))
    for index, (dataset, block) in enumerate(zip(DATASETS, blocks)):
        points = coordinates[labels == index]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=dataset["color"],
            s=42,
            alpha=0.8,
            linewidths=0.6,
            edgecolors="#FCFCFB",
        )

    ax.set(
        title=f"{MODEL['name']} mean-pooled protein-embedding UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )
    # UMAP axes have no units, so their tick values carry no meaning.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.99, 0.01,
        f"{N_NEIGHBORS} neighbours · min_dist {MIN_DIST} · "
        f"{PCA_COMPONENTS} PCs · seed {RANDOM_STATE}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#333333",
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=dataset["color"],
                label=f"{dataset['name']} ({block.shape[0]:,} TFs)",
            )
            for dataset, block in zip(DATASETS, blocks)
        ],
        frameon=False,
        loc="best",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_mean_pooled_protein_umap.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    print(f"Protein vectors: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"UMAP: {PCA_COMPONENTS} PCs, n_neighbors {N_NEIGHBORS}, "
          f"min_dist {MIN_DIST}, seed {RANDOM_STATE}")
    print(f"Trustworthiness ({TRUST_NEIGHBOURS} neighbours): {trust:.3f}")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
