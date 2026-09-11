"""Plot mean-pooled ESMC-600M protein PCA at every second layer in one figure.

Reads the pooled vectors from embedding/extract_layer_pooled_embeddings.py and
fits a separate exact PCA per layer. Layers 2-34 are the un-normalised
residual stream, whose scale grows with depth; layer 36 is after the final
LayerNorm and matches plot_pooled_embedding_pca.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M"}
DATASETS = (
    {"key": "ecoli", "name": "E. coli K-12", "color": "#0072FF"},
    {"key": "pao1", "name": "P. aeruginosa PAO1", "color": "#FF6B00"},
    {"key": "hvolc", "name": "H. volcanii DS2", "color": "#00A6A6"},
    {"key": "nrc1", "name": "H. salinarum NRC-1", "color": "#7B3FF2"},
)
LAYER_NOTES = {0: "token embedding", 36: "final, after LayerNorm"}


def pca_2d(matrix, labels):
    """Exact PCA; PC1 puts E. coli on the left, PC2 puts H. volcanii on top."""
    centered = matrix - matrix.mean(dim=0)
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    explained = singular_values[:2].square() / singular_values.square().sum()
    if coordinates[labels == 0, 0].mean() > 0:
        coordinates[:, 0] *= -1
    if coordinates[labels == 2, 1].mean() < 0:
        coordinates[:, 1] *= -1
    return coordinates, explained


def main():
    source = torch.load(
        PROJECT_DIR / "embedding" / MODEL["slug"]
        / f"{MODEL['slug']}_layer_pooled_tf_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    layers = source["layers"]
    blocks = [source["datasets"][dataset["key"]]["pooled"] for dataset in DATASETS]
    stacked = torch.cat(blocks).double()
    labels = torch.cat([
        torch.full((block.shape[0],), index) for index, block in enumerate(blocks)
    ])

    columns = 3
    rows = -(-len(layers) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.7 * rows))
    print(f"Model: {MODEL['name']}; protein vectors: {stacked.shape[0]} x {stacked.shape[2]}")
    for ax, (position, layer) in zip(axes.flat, enumerate(layers)):
        coordinates, explained = pca_2d(stacked[:, position], labels)
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
        note = f" ({LAYER_NOTES[layer]})" if layer in LAYER_NOTES else ""
        ax.set_title(f"Layer {layer}{note}", fontweight="bold")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        print(f"  layer {layer:>2}: PC1 {explained[0] * 100:5.1f}%  "
              f"PC2 {explained[1] * 100:5.1f}%  "
              f"mean vector norm {stacked[:, position].norm(dim=1).mean():8.2f}")
    for ax in axes.flat[len(layers):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{MODEL['name']} mean-pooled protein-embedding PCA by layer",
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
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = (
        PROJECT_DIR
        / "results"
        / "dimensionality_reduction"
        / f"{MODEL['name']}_mean_pooled_protein_pca_by_layer.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
