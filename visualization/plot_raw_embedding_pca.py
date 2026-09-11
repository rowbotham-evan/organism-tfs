"""Plot a raw residue-level ESMC-600M PCA across all three TF datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
# Categorical hues in fixed order; validated for CVD separation against the
# plot surface (worst adjacent pair dE 32.6 protan / 29.5 tritan).
DATASETS = (
    {
        "key": "ecoli",
        "name": "E. coli K-12",
        "suffix": "all_raw_embeddings.pt",
        "color": "#0072FF",
    },
    {
        "key": "pao1",
        "name": "P. aeruginosa PAO1",
        "suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "color": "#FF6B00",
    },
    {
        "key": "hvolc",
        "name": "H. volcanii DS2",
        "suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "color": "#00A6A6",
    },
    {
        "key": "nrc1",
        "name": "H. salinarum NRC-1",
        "suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "color": "#7B3FF2",
    },
)


def load_raw_embeddings(dataset):
    path = (
        PROJECT_DIR
        / "embedding"
        / MODEL["slug"]
        / f"{MODEL['slug']}_{dataset['suffix']}"
    )
    records = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = []
    for record in records:
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(record["embedding"].shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, "
                f"got {tuple(record['embedding'].shape)}"
            )
        embeddings.append(record["embedding"].float())
    return torch.cat(embeddings), len(records)


def main():
    blocks, counts = [], []
    for dataset in DATASETS:
        residues, proteins = load_raw_embeddings(dataset)
        blocks.append(residues)
        counts.append({"proteins": proteins, "residues": residues.shape[0]})

    matrix = torch.cat(blocks)
    del blocks

    fit_mean = matrix.mean(dim=0)
    centered = matrix - fit_mean
    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=2,
        center=False,
        niter=4,
    )
    coordinates = centered @ components[:, :2]
    explained = singular_values[:2].square() / centered.square().sum()
    del centered

    # Split the projected rows back out per dataset, in load order.
    offsets, start = [], 0
    for count in counts:
        offsets.append((start, start + count["residues"]))
        start += count["residues"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # All datasets are drawn interleaved in one randomised pass so no organism
    # sits systematically on top of another.
    order = torch.cat([torch.arange(begin, end) for begin, end in offsets])
    colors = []
    for dataset, (begin, end) in zip(DATASETS, offsets):
        colors.extend([dataset["color"]] * (end - begin))
    generator = torch.Generator().manual_seed(7)
    shuffle = torch.randperm(order.shape[0], generator=generator)
    order = order[shuffle]
    colors = [colors[index] for index in shuffle.tolist()]
    ax.scatter(
        coordinates[order, 0], coordinates[order, 1],
        c=colors, s=2, alpha=0.12, linewidths=0, rasterized=True,
    )

    ax.set(
        title=f"{MODEL['name']} raw residue-embedding PCA",
        xlabel=f"PC1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}% variance)",
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=dataset["color"],
                label=(
                    f"{dataset['name']} "
                    f"({count['proteins']:,} TFs, {count['residues']:,} residues)"
                ),
            )
            for dataset, count in zip(DATASETS, counts)
        ],
        frameon=False,
        loc="best",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output = (
        PROJECT_DIR
        / "results"
        / "dimensionality_reduction"
        / f"{MODEL['name']}_raw_residue_pca_all_organisms.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    for dataset, count in zip(DATASETS, counts):
        print(f"  {dataset['name']}: {count['proteins']} TFs, "
              f"{count['residues']:,} residues")
    print(f"Raw residue rows: {matrix.shape[0]:,}")
    print(f"Embedding width: {matrix.shape[1]}")
    print(f"PC1 + PC2 variance: {explained.sum() * 100:.1f}%")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
