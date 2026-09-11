"""Plot mean-pooled ESMC-600M t-SNE at increasing perplexity in one grid figure.

Pooling, the 50-PC reduction and all other t-SNE settings match
plot_pooled_embedding_tsne.py; only perplexity changes between panels.
Perplexity is roughly the number of neighbours each point attends to, so low
values emphasise very local structure and high values more global structure.
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from plot_pooled_embedding_tsne import (
    DATASETS,
    MODEL,
    PCA_COMPONENTS,
    PROJECT_DIR,
    RANDOM_STATE,
    pool_dataset,
)


PERPLEXITIES = tuple(range(5, 51, 5))
COLUMNS = 5


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    matrix = torch.cat(blocks).numpy().astype(np.float64)
    labels = np.concatenate([
        np.full(block.shape[0], index) for index, block in enumerate(blocks)
    ])
    reduced = PCA(
        n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
    ).fit_transform(matrix)

    rows = -(-len(PERPLEXITIES) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(5 * COLUMNS, 4.7 * rows))
    print(f"Model: {MODEL['name']}")
    print(f"Protein vectors: {matrix.shape[0]} x {matrix.shape[1]}")
    for ax, perplexity in zip(axes.flat, PERPLEXITIES):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        coordinates = tsne.fit_transform(reduced)
        print(f"  perplexity {perplexity:>2}: KL divergence "
              f"{tsne.kl_divergence_:.3f}")

        for index, dataset in enumerate(DATASETS):
            points = coordinates[labels == index]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=dataset["color"],
                s=12,
                alpha=0.8,
                linewidths=0.3,
                edgecolors="#FCFCFB",
            )
        ax.set_title(f"Perplexity {perplexity}", fontweight="bold")
        # t-SNE axes have no units, so their tick values carry no meaning.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

    # Label the lowest used panel in each column; hide panels past the sweep.
    for position, ax in enumerate(axes.flat):
        if position >= len(PERPLEXITIES):
            ax.set_visible(False)
        elif position + COLUMNS >= len(PERPLEXITIES):
            ax.set_xlabel("t-SNE 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("t-SNE 2")

    fig.suptitle(
        f"{MODEL['name']} mean-pooled protein-embedding t-SNE by perplexity",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=dataset["color"],
                label=f"{dataset['name']} ({block.shape[0]:,} TFs)",
            )
            for dataset, block in zip(DATASETS, blocks)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(DATASETS),
        frameon=False,
    )
    fig.text(
        0.99, 0.005,
        f"{PCA_COMPONENTS} PCs · seed {RANDOM_STATE}",
        ha="right", va="bottom", fontsize=9, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))

    output = (
        PROJECT_DIR
        / "results"
        / "dimensionality_reduction"
        / (f"{MODEL['name']}_mean_pooled_protein_tsne_perplexity_"
           f"{PERPLEXITIES[0]}_to_{PERPLEXITIES[-1]}.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
