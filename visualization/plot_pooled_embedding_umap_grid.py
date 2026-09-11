"""Plot mean-pooled ESMC-600M UMAP over n_neighbors and min_dist in one grid.

Inputs and every other setting match plot_pooled_embedding_umap.py. Columns
raise n_neighbors (how many neighbours define local structure; higher values
favour global layout) and rows raise min_dist (how tightly points may pack in
2-d; it changes the look, not the neighbour graph). Each panel reports
trustworthiness against the original 1152-d vectors.
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import torch
import umap

from plot_pooled_embedding_umap import (
    DATASETS,
    MODEL,
    PCA_COMPONENTS,
    PROJECT_DIR,
    RANDOM_STATE,
    TRUST_NEIGHBOURS,
    pool_dataset,
)


N_NEIGHBORS = tuple(range(5, 51, 5))
MIN_DISTS = (0.0, 0.1, 0.5)


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    matrix = torch.cat(blocks).numpy().astype(np.float64)
    labels = np.concatenate([
        np.full(block.shape[0], index) for index, block in enumerate(blocks)
    ])
    reduced = PCA(
        n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
    ).fit_transform(matrix)

    fig, axes = plt.subplots(
        len(MIN_DISTS),
        len(N_NEIGHBORS),
        figsize=(3.2 * len(N_NEIGHBORS), 3.2 * len(MIN_DISTS)),
        squeeze=False,
    )
    print(f"Model: {MODEL['name']}; protein vectors: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Trustworthiness ({TRUST_NEIGHBOURS} neighbours):")
    for row, min_dist in enumerate(MIN_DISTS):
        scores = []
        for column, n_neighbors in enumerate(N_NEIGHBORS):
            ax = axes[row, column]
            coordinates = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=RANDOM_STATE,
            ).fit_transform(reduced)
            trust = trustworthiness(matrix, coordinates, n_neighbors=TRUST_NEIGHBOURS)
            scores.append(trust)

            for index, dataset in enumerate(DATASETS):
                points = coordinates[labels == index]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=dataset["color"],
                    s=6,
                    alpha=0.8,
                    linewidths=0.2,
                    edgecolors="#FCFCFB",
                )
            # UMAP axes have no units, so their tick values carry no meaning.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                0.98, 0.02, f"trust {trust:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#333333",
            )
            if row == 0:
                ax.set_title(f"n_neighbors {n_neighbors}", fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"min_dist {min_dist}", fontweight="bold", fontsize=11)
        print(f"  min_dist {min_dist}: " + "  ".join(
            f"n{n_neighbors}={score:.3f}"
            for n_neighbors, score in zip(N_NEIGHBORS, scores)
        ))

    fig.suptitle(
        f"{MODEL['name']} mean-pooled protein-embedding UMAP by n_neighbors and min_dist",
        fontsize=16,
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
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(DATASETS),
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.995, 0.005,
        f"{PCA_COMPONENTS} PCs · seed {RANDOM_STATE} · "
        f"trustworthiness on {TRUST_NEIGHBOURS} neighbours",
        ha="right", va="bottom", fontsize=9, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.91))

    output = (
        PROJECT_DIR
        / "results"
        / "dimensionality_reduction"
        / f"{MODEL['name']}_mean_pooled_protein_umap_grid.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
