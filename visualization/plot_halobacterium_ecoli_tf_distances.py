"""Plot each H. salinarum NRC-1 TF's distance to its closest E. coli TF.

Each TF is reduced to one mean-pooled ESMC-600M vector, then every NRC-1 TF is
matched to the nearest E. coli K-12 TF by Euclidean distance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
BIN_WIDTH = 0.01
INK = "#333333"
COLOR = "#7B3FF2"
LEFT = {
    "name": "H. salinarum NRC-1",
    "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
}
RIGHT = {
    "name": "E. coli K-12",
    "suffix": "all_raw_embeddings.pt",
}


def count_box(axis, text):
    axis.text(
        0.995, 0.96, text,
        transform=axis.transAxes, ha="right", va="top",
        fontsize=9, color=INK,
    )


def line_legend(axis, mean_value, median_value, fmt="{:.3f}"):
    """Mean and median values live in the side legend, not on the bars."""
    axis.legend(
        handles=(
            Line2D([], [], color=INK, linewidth=1.6,
                   label=f"Mean {fmt.format(mean_value)}"),
            Line2D([], [], color=INK, linewidth=1.6, linestyle="--",
                   label=f"Median {fmt.format(median_value)}"),
        ),
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.90),
        fontsize=9,
    )


def pooled_vectors(dataset):
    records = torch.load(
        PROJECT_DIR / "embedding" / MODEL["slug"]
        / f"{MODEL['slug']}_{dataset['suffix']}",
        map_location="cpu",
        weights_only=False,
    )

    pooled, genes = [], []
    for record in records:
        raw = record["embedding"].double()
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, got {tuple(raw.shape)}"
            )
        pooled.append(raw.mean(dim=0))
        genes.append(record["gene"])
    return torch.stack(pooled), genes


def main():
    left, left_genes = pooled_vectors(LEFT)
    right, right_genes = pooled_vectors(RIGHT)

    matrix = torch.cdist(left, right)          # every NRC-1 TF x every E. coli TF
    nearest, partner = matrix.min(dim=1)
    distances = nearest.numpy()

    bins = np.arange(
        np.floor(distances.min() / BIN_WIDTH) * BIN_WIDTH,
        (np.ceil(distances.max() / BIN_WIDTH) + 1) * BIN_WIDTH,
        BIN_WIDTH,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=bins, color=COLOR, edgecolor="white", linewidth=0.6)
    ax.axvline(distances.mean(), color=INK, linewidth=1.6)
    ax.axvline(np.median(distances), color=INK, linewidth=1.6, linestyle="--")
    ax.set_title(
        f"Nearest {RIGHT['name']} TF for each {LEFT['name']} TF",
        loc="center",
        fontweight="bold",
    )
    count_box(ax, f"n = {distances.size}")
    line_legend(ax, distances.mean(), np.median(distances))
    ax.set(
        xlabel=f"Euclidean distance to nearest {RIGHT['name']} TF",
        ylabel="Number of transcription factors",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_nrc1_to_ecoli_tf_distances.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    flat = np.sort(distances)
    q = lambda p: flat[int(p * (len(flat) - 1))]
    print(f"Model: {MODEL['name']}")
    print(f"{LEFT['name']} ({len(left_genes)} TFs) -> nearest of "
          f"{RIGHT['name']} ({len(right_genes)} TFs)")
    print(f"  min {flat.min():.4f}  p25 {q(.25):.4f}  median {q(.50):.4f}  "
          f"p75 {q(.75):.4f}  max {flat.max():.4f}  sd {distances.std():.4f}")
    order = np.argsort(distances)
    print("\n  closest NRC-1 -> E. coli pairs:")
    for i in order[:6]:
        print(f"    {left_genes[i]:<12} -> {right_genes[partner[i]]:<12} "
              f"{distances[i]:.4f}")
    print("\n  most isolated NRC-1 TFs:")
    for i in order[-3:]:
        print(f"    {left_genes[i]:<12} -> {right_genes[partner[i]]:<12} "
              f"{distances[i]:.4f}")
    print(f"\nSaved plot: {output}")


if __name__ == "__main__":
    main()
