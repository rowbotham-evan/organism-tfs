"""Plot each TF's distance to its nearest other TF within the same organism.

Every TF is reduced to one mean-pooled ESMC-600M vector, then for each TF the
Euclidean distance to the closest other TF in the same organism is taken
(self excluded). Tight neighbours are typically paralogs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
BIN_WIDTH = 0.02
INK = "#333333"
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
)


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


def nearest_neighbour_distances(dataset):
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

    matrix = torch.stack(pooled)
    distances = torch.cdist(matrix, matrix)
    distances.fill_diagonal_(float("inf"))
    nearest, partner = distances.min(dim=1)
    return nearest.numpy(), partner.numpy(), genes


def main():
    results = [nearest_neighbour_distances(dataset) for dataset in DATASETS]
    upper = max(nearest.max() for nearest, _, _ in results)
    bins = np.arange(0, (np.ceil(upper / BIN_WIDTH) + 1) * BIN_WIDTH, BIN_WIDTH)

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(10, 9), sharex=True)

    for axis, dataset, (nearest, partner, genes) in zip(axes, DATASETS, results):
        axis.hist(
            nearest,
            bins=bins,
            color=dataset["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.axvline(nearest.mean(), color=INK, linewidth=1.6)
        axis.axvline(np.median(nearest), color=INK, linewidth=1.6, linestyle="--")
        axis.set_title(dataset["name"], loc="center", fontweight="bold")
        count_box(axis, f"n = {len(nearest)}")
        line_legend(axis, nearest.mean(), np.median(nearest))
        axis.set_ylabel("TFs")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)

        # Name the single most isolated TF; it is the tail the axis is showing.
        far = int(nearest.argmax())
        axis.annotate(
            f"{genes[far]} ({nearest[far]:.2f})",
            xy=(nearest[far], 1),
            xytext=(nearest[far], axis.get_ylim()[1] * 0.45),
            ha="center",
            fontsize=8,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8),
        )

    axes[-1].set_xlabel("Euclidean distance to nearest other self TF")
    fig.suptitle(
        f"{MODEL['name']} nearest-neighbour distance between TFs",
        x=0.5,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_tf_nearest_neighbour_distances.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    for dataset, (nearest, partner, genes) in zip(DATASETS, results):
        order = nearest.argsort()
        closest = ", ".join(
            f"{genes[i]}-{genes[partner[i]]} {nearest[i]:.3f}" for i in order[:3:2]
        )
        print(f"\n  {dataset['name']}  (n={len(nearest)})")
        print(f"    min {nearest.min():.4f}  median {np.median(nearest):.4f}  "
              f"mean {nearest.mean():.4f}  max {nearest.max():.4f}")
        print(f"    closest pairs: {closest}")
        print(f"    most isolated: {genes[order[-1]]} ({nearest[order[-1]]:.4f})")
    print(f"\nSaved plot: {output}")


if __name__ == "__main__":
    main()
