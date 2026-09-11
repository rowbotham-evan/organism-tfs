"""Plot a protein-level PCA of mean-pooled ESMC-600M embeddings.

Pooling matches the standard ESM recipe: take the final-layer residue
representations with the special tokens excluded, then average over the
residue axis. The reference implementation slices `[1 : truncate_len + 1]`
to drop BOS and stop before EOS; `encode_all_tfs.py` already strips special
tokens with an attention/special-token mask, so a plain mean over dim 0 of
the stored tensors is the same vector. No L2 normalization is applied.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
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
        raw = record["embedding"].float()
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, got {tuple(raw.shape)}"
            )
        pooled.append(raw.mean(dim=0))
    return torch.stack(pooled)


def main():
    blocks = [pool_dataset(dataset) for dataset in DATASETS]
    matrix = torch.cat(blocks)

    fit_mean = matrix.mean(dim=0)
    centered = matrix - fit_mean
    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=2,
        center=False,
        niter=7,
    )
    coordinates = centered @ components[:, :2]
    explained = singular_values[:2].square() / centered.square().sum()

    fig, ax = plt.subplots(figsize=(9, 8))
    start = 0
    for dataset, block in zip(DATASETS, blocks):
        end = start + block.shape[0]
        ax.scatter(
            coordinates[start:end, 0],
            coordinates[start:end, 1],
            c=dataset["color"],
            s=42,
            alpha=0.8,
            linewidths=0.6,
            edgecolors="#FCFCFB",
        )
        start = end

    ax.set(
        title=f"{MODEL['name']} mean-pooled protein-embedding PCA",
        xlabel=f"PC1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}% variance)",
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=dataset["color"],
                label=f"{dataset['name']} ({block.shape[0]:,} TFs)",
            )
            for dataset, block in zip(DATASETS, blocks)
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
        / f"{MODEL['name']}_mean_pooled_protein_pca.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {MODEL['name']}")
    for dataset, block in zip(DATASETS, blocks):
        print(f"  {dataset['name']}: {block.shape[0]} proteins")
    print(f"Protein vectors: {matrix.shape[0]}")
    print(f"Embedding width: {matrix.shape[1]}")
    print(f"PC1 + PC2 variance: {explained.sum() * 100:.1f}%")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
