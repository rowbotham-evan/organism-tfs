"""Plot higher principal components of the mean-pooled ESMC-600M embeddings.

Uses the same per-TF vectors as plot_pooled_embedding_pca.py, but computes the
first eight principal components with an exact SVD and draws them in one
figure with a panel per new component: PC2-PC3, PC3-PC4, ... PC7-PC8. Each
panel adds exactly one component beyond the PC1-PC2 plot.

For every component the script also reports how much of its variance is
explained by organism, and by domain of life (bacteria vs archaea), so it is
clear which components carry the separations visible in the plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
COMPONENTS = 8
FIRST_NEW = 3
DATASETS = (
    {
        "name": "E. coli K-12",
        "suffix": "all_raw_embeddings.pt",
        "color": "#0072FF",
        "domain": "bacteria",
    },
    {
        "name": "P. aeruginosa PAO1",
        "suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "color": "#FF6B00",
        "domain": "bacteria",
    },
    {
        "name": "H. volcanii DS2",
        "suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "color": "#00A6A6",
        "domain": "archaea",
    },
    {
        "name": "H. salinarum NRC-1",
        "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "color": "#7B3FF2",
        "domain": "archaea",
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
        raw = record["embedding"].double()
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, got {tuple(raw.shape)}"
            )
        pooled.append(raw.mean(dim=0))
    return torch.stack(pooled)


def principal_components(matrix):
    centered = matrix - matrix.mean(dim=0)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    components = right[:COMPONENTS]
    # SVD signs are arbitrary; make each component's largest loading positive
    # so the plots do not mirror between runs.
    signs = torch.sign(components[torch.arange(COMPONENTS), components.abs().argmax(dim=1)])
    components = components * signs[:, None]
    scores = centered @ components.T
    explained = singular_values.square() / singular_values.square().sum()
    return scores.numpy(), explained[:COMPONENTS].numpy()


def variance_share(values, groups):
    """Fraction of a component's variance explained by group membership (R^2)."""
    total = ((values - values.mean()) ** 2).sum()
    between = sum(
        (groups == group).sum() * (values[groups == group].mean() - values.mean()) ** 2
        for group in np.unique(groups)
    )
    return between / total


def plot_pairs(scores, explained, labels, counts, pairs):
    """All component pairs as panels of one figure, sharing a single legend."""
    columns = 3
    rows = -(-len(pairs) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 5.6 * rows))
    for axis, (first, second) in zip(axes.flat, pairs):
        for index, dataset in enumerate(DATASETS):
            points = scores[labels == index]
            axis.scatter(
                points[:, first - 1],
                points[:, second - 1],
                c=dataset["color"],
                s=18,
                alpha=0.75,
                linewidths=0.4,
                edgecolors="#FCFCFB",
            )
        axis.set_title(f"PC{first} vs PC{second}", fontweight="bold")
        axis.set_xlabel(f"PC{first} ({explained[first - 1] * 100:.1f}% variance)")
        axis.set_ylabel(f"PC{second} ({explained[second - 1] * 100:.1f}% variance)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(pairs):]:
        axis.set_visible(False)

    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=dataset["color"],
                label=f"{dataset['name']} ({count:,} TFs)",
            )
            for dataset, count in zip(DATASETS, counts)
        ],
        loc="upper center",
        ncol=len(DATASETS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        f"{MODEL['name']} mean-pooled protein-embedding PCA: components 2-8",
        y=1.03,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = (
        PROJECT_DIR
        / "results"
        / "dimensionality_reduction"
        / f"{MODEL['name']}_mean_pooled_protein_pca_pc2_to_pc8.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    counts = [block.shape[0] for block in blocks]
    matrix = torch.cat(blocks)
    labels = np.concatenate([
        np.full(count, index) for index, count in enumerate(counts)
    ])
    domains = np.array([DATASETS[index]["domain"] for index in labels])

    scores, explained = principal_components(matrix)

    print(f"Model: {MODEL['name']}   protein vectors: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"{'PC':<5}{'variance':>10}{'cumulative':>12}{'by organism':>13}{'by domain':>11}")
    cumulative = 0.0
    for component in range(COMPONENTS):
        cumulative += explained[component]
        values = scores[:, component]
        print(f"PC{component + 1:<3}{explained[component] * 100:>9.1f}%{cumulative * 100:>11.1f}%"
              f"{variance_share(values, labels) * 100:>12.1f}%{variance_share(values, domains) * 100:>10.1f}%")

    pairs = [(second - 1, second) for second in range(FIRST_NEW, COMPONENTS + 1)]
    output = plot_pairs(scores, explained, labels, counts, pairs)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
