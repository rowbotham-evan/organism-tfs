"""Plot TF sequence-length distributions for all datasets in one panel."""

from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
BIN_WIDTH = 50
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
# Same hue per organism as the PCA: colour follows the entity across charts.
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
INK = "#333333"


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


def load_lengths(dataset):
    proteins = torch.load(
        PROJECT_DIR / "embedding" / MODEL["slug"]
        / f"{MODEL['slug']}_{dataset['suffix']}",
        map_location="cpu",
        weights_only=False,
    )

    lengths = []
    for protein in proteins:
        length = len(protein["sequence"])
        expected_shape = (length, MODEL["width"])
        if tuple(protein["embedding"].shape) != expected_shape:
            raise ValueError(
                f"{protein['gene']}: expected {expected_shape}, "
                f"got {tuple(protein['embedding'].shape)}"
            )
        lengths.append(length)

    return lengths


def main():
    lengths = [load_lengths(dataset) for dataset in DATASETS]
    # Shared x so the three distributions are directly comparable; counts differ
    # by an order of magnitude, so each panel keeps its own y.
    upper_bound = ((max(max(group) for group in lengths) // BIN_WIDTH) + 1) * BIN_WIDTH
    bins = range(0, upper_bound + BIN_WIDTH, BIN_WIDTH)

    fig, axes = plt.subplots(
        len(DATASETS), 1, figsize=(10, 9), sharex=True,
    )

    for axis, dataset, group in zip(axes, DATASETS, lengths):
        mean_length = mean(group)
        median_length = median(group)
        axis.hist(
            group,
            bins=bins,
            color=dataset["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.axvline(mean_length, color=INK, linewidth=1.6)
        axis.axvline(median_length, color=INK, linewidth=1.6, linestyle="--")
        axis.set_title(dataset["name"], loc="center", fontweight="bold")
        count_box(axis, f"n = {len(group)}")
        line_legend(axis, mean_length, median_length, fmt="{:.0f} aa")
        axis.set_ylabel("Number of transcription factors")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("Protein length (amino acids)")
    fig.suptitle(
        f"{MODEL['name']} transcription-factor sequence lengths",
        x=0.5,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = (
        PROJECT_DIR
        / "results"
        / f"{MODEL['name']}_tf_sequence_length_distributions.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    for dataset, group in zip(DATASETS, lengths):
        print(f"  {dataset['name']}: n={len(group)}, "
              f"range {min(group)}-{max(group)} aa, "
              f"mean {mean(group):.1f}, median {median(group):.0f}")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
