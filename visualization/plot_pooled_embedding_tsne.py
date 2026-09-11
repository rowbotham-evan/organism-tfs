"""Plot a protein-level t-SNE of mean-pooled ESMC-600M embeddings.

Pooling is the same as plot_pooled_embedding_pca.py: one vector per TF, the
mean of its residue embeddings. The vectors are reduced to 50 principal
components before t-SNE, the usual step to remove noise and speed it up.

t-SNE keeps local neighbourhoods but not global geometry: cluster sizes and
the gaps between clusters are not interpretable as distances. The script
therefore also reports, in the original 1152-d space, how often each TF's
nearest neighbours come from its own organism.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
PCA_COMPONENTS = 50
PERPLEXITY = 30
RANDOM_STATE = 0
NEIGHBOURS = 10
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


def neighbour_purity(matrix, labels):
    """Share of each TF's nearest neighbours (self excluded) from its own organism."""
    finder = NearestNeighbors(n_neighbors=NEIGHBOURS + 1).fit(matrix)
    _, indices = finder.kneighbors(matrix)
    same = labels[indices[:, 1:]] == labels[:, None]
    return same.mean(axis=1)


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    matrix = torch.cat(blocks).numpy().astype(np.float64)
    labels = np.concatenate([
        np.full(block.shape[0], index) for index, block in enumerate(blocks)
    ])

    reduced = PCA(
        n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
    ).fit_transform(matrix)
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    coordinates = tsne.fit_transform(reduced)
    purity = neighbour_purity(matrix, labels)

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
        title=f"{MODEL['name']} mean-pooled protein-embedding t-SNE",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    # t-SNE axes have no units, so their tick values carry no meaning.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.99, 0.01,
        f"perplexity {PERPLEXITY} · {PCA_COMPONENTS} PCs · seed {RANDOM_STATE}",
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
        / f"{MODEL['name']}_mean_pooled_protein_tsne.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    print(f"Protein vectors: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"t-SNE: {PCA_COMPONENTS} PCs, perplexity {PERPLEXITY}, "
          f"seed {RANDOM_STATE}, KL divergence {tsne.kl_divergence_:.3f}")
    print(f"Same-organism share of {NEIGHBOURS} nearest neighbours "
          f"(original {matrix.shape[1]}-d space):")
    for index, dataset in enumerate(DATASETS):
        print(f"  {dataset['name']:<20} {purity[labels == index].mean():.1%}")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
