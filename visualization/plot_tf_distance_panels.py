"""Plot each organism's nearest-TF distances to itself and every other organism.

Every TF is reduced to one mean-pooled ESMC-600M vector. For one query
organism, each of its TFs is matched to the closest TF in a target organism by
Euclidean distance; when the target is the query organism itself the TF is
excluded, so that panel is the distance to the nearest other self TF. One
figure per query organism stacks the four targets on a shared distance axis,
self first. Bars take the target organism's colour.
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
    "nrc1": {
        "name": "H. salinarum NRC-1",
        "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "color": "#7B3FF2",
    },
    "hvolc": {
        "name": "H. volcanii DS2",
        "suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "color": "#00A6A6",
    },
}


def count_label(axis, text):
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


def nearest_distances(query, target, same_organism):
    distances = torch.cdist(query, target)
    if same_organism:
        distances.fill_diagonal_(float("inf"))
    nearest, partner = distances.min(dim=1)
    return nearest.numpy(), partner.numpy()


def plot_query(query_key, vectors):
    query_set = DATASETS[query_key]
    query, query_genes = vectors[query_key]
    targets = [query_key] + [key for key in DATASETS if key != query_key]
    results = [
        nearest_distances(query, vectors[key][0], key == query_key)
        for key in targets
    ]

    upper = max(nearest.max() for nearest, _ in results)
    bins = np.arange(0, (np.ceil(upper / BIN_WIDTH) + 1) * BIN_WIDTH, BIN_WIDTH)
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, 14), sharex=True)

    print(f"\n{query_set['name']} (n={len(query_genes)})")
    for axis, target_key, (nearest, partner) in zip(axes, targets, results):
        target_set = DATASETS[target_key]
        same = target_key == query_key
        axis.hist(
            nearest,
            bins=bins,
            color=target_set["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.axvline(nearest.mean(), color=INK, linewidth=1.6)
        axis.axvline(np.median(nearest), color=INK, linewidth=1.6, linestyle="--")
        axis.set_title(
            f"{query_set['name']} vs {target_set['name']}",
            loc="center",
            fontweight="bold",
        )
        axis.set_xlabel(
            "Euclidean distance to nearest other self TF" if same
            else f"Euclidean distance to nearest {target_set['name']} TF"
        )
        axis.set_ylabel("Number of transcription factors")
        axis.tick_params(labelbottom=True)
        count_label(axis, f"n = {len(nearest)}")
        line_legend(axis, nearest.mean(), np.median(nearest))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)

        target_genes = vectors[target_key][1]
        far = int(nearest.argmax())
        print(f"  vs {target_set['name']:<20} min {nearest.min():.4f}  "
              f"median {np.median(nearest):.4f}  mean {nearest.mean():.4f}  "
              f"max {nearest.max():.4f}  (most isolated: {query_genes[far]} -> "
              f"{target_genes[partner[far]]})")

    fig.suptitle(
        f"{MODEL['name']} nearest-TF distances for each {query_set['name']} TF",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_{query_key}_tf_distances.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output}")


def main():
    vectors = {key: pooled_vectors(dataset) for key, dataset in DATASETS.items()}
    print(f"Model: {MODEL['name']}")
    for key in DATASETS:
        plot_query(key, vectors)


if __name__ == "__main__":
    main()
