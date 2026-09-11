"""Plot each query-organism TF's distance to its closest target-organism TF.

Both organisms' TFs are reduced to one mean-pooled ESMC-600M vector each,
then every query TF is matched to the nearest target TF by Euclidean
distance. Small distances mark TFs with a close counterpart in the other
organism.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
BIN_WIDTH = 0.02
INK = "#333333"
DATASETS = {
    "ecoli": {
        "name": "E. coli K-12",
        "suffix": "all_raw_embeddings.pt",
        "color": "#0072FF",
    },
    "pao1": {
        "name": "P. aeruginosa PAO1",
        "suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "color": "#FF6B00",
    },
    "hvolc": {
        "name": "H. volcanii DS2",
        "suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "color": "#00A6A6",
    },
    "nrc1": {
        "name": "H. salinarum NRC-1",
        "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "color": "#7B3FF2",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", choices=DATASETS, default="pao1")
    parser.add_argument("--target", choices=DATASETS, default="ecoli")
    return parser.parse_args()


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
    args = parse_args()
    if args.query == args.target:
        raise ValueError("Choose two different organisms")
    query_set, target_set = DATASETS[args.query], DATASETS[args.target]
    query, query_genes = pooled_vectors(query_set)
    target, target_genes = pooled_vectors(target_set)

    distances = torch.cdist(query, target)
    nearest, partner = distances.min(dim=1)
    nearest = nearest.numpy()
    partner = partner.numpy()

    bins = np.arange(
        0, (np.ceil(nearest.max() / BIN_WIDTH) + 1) * BIN_WIDTH, BIN_WIDTH
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nearest, bins=bins, color=query_set["color"], edgecolor="white", linewidth=0.8)
    ax.axvline(nearest.mean(), color=INK, linewidth=1.6)
    ax.axvline(np.median(nearest), color=INK, linewidth=1.6, linestyle="--")

    ax.set(
        title=(
            f"Each {query_set['name']} TF to its closest {target_set['name']} TF "
            f"({MODEL['name']}, mean-pooled)"
        ),
        xlabel=f"Euclidean distance to nearest {target_set['name']} TF",
        ylabel="Number of transcription factors",
    )
    count_box(ax, f"n = {len(nearest)}")
    line_legend(ax, nearest.mean(), np.median(nearest))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_{args.query}_to_{args.target}_tf_distances.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    order = nearest.argsort()
    print(f"Model: {MODEL['name']}")
    print(f"{query_set['name']} TFs: {len(query_genes)}  ->  "
          f"{target_set['name']} TFs: {len(target_genes)}")
    print(f"  min {nearest.min():.4f}  median {np.median(nearest):.4f}  "
          f"mean {nearest.mean():.4f}  max {nearest.max():.4f}")
    print("\n  closest cross-organism pairs:")
    for i in order[:6]:
        print(f"    {query_genes[i]:<12} -> {target_genes[partner[i]]:<12} "
              f"{nearest[i]:.4f}")
    print(f"\n  most isolated {query_set['name']} TFs:")
    for i in order[-3:]:
        print(f"    {query_genes[i]:<12} -> {target_genes[partner[i]]:<12} "
              f"{nearest[i]:.4f}")
    print(f"\nSaved plot: {output}")


if __name__ == "__main__":
    main()
