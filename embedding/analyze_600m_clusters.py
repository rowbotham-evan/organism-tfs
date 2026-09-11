"""Cluster statistics for each organism's TF embeddings, written to CSV.

A cluster is one organism's set of TF vectors. Two spaces are reported:

  pooled   one vector per TF, the mean of its residue embeddings
  raw      one vector per residue, no pooling

For each cluster: the mean vector, its norm, and the RMS distance of members
about it. For each pair: cosine similarity and Euclidean distance between the
mean vectors, alongside the bounds those measures can reach.
"""

import csv
from itertools import combinations
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152}
DATASETS = {
    "ecoli": {
        "name": "E. coli K-12",
        "source_suffix": "all_raw_embeddings.pt",
        "output_slug": "ecoli_k12",
    },
    "pao1": {
        "name": "P. aeruginosa PAO1",
        "source_suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "output_slug": "pseudomonas_aeruginosa_pao1",
    },
    "hvolc": {
        "name": "H. volcanii DS2",
        "source_suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "output_slug": "haloferax_volcanii_ds2",
    },
    "nrc1": {
        "name": "H. salinarum NRC-1",
        "source_suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "output_slug": "halobacterium_salinarum_nrc1",
    },
}


def model_dir():
    return PROJECT_DIR / "embedding" / MODEL["slug"]


def load_spaces(dataset):
    """Return (pooled protein vectors, raw residue vectors) for one dataset."""
    records = torch.load(
        model_dir() / f"{MODEL['slug']}_{dataset['source_suffix']}",
        map_location="cpu",
        weights_only=False,
    )

    pooled, residues = [], []
    for record in records:
        matrix = record["embedding"].double()
        expected = (len(record["sequence"]), MODEL["width"])
        if tuple(matrix.shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, got {tuple(matrix.shape)}"
            )
        pooled.append(matrix.mean(dim=0))
        residues.append(matrix)

    if not pooled:
        raise ValueError(f"{dataset['name']}: no proteins")
    return torch.stack(pooled), torch.cat(residues)


def save_mean_vector(dataset, mean_vector, tf_count):
    output = (
        model_dir()
        / f"{MODEL['slug']}_{dataset['output_slug']}_mean_vector.csv"
    )
    with output.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "model", "organism", "tf_count",
            *(f"embedding_{index}" for index in range(1, MODEL["width"] + 1)),
        ])
        writer.writerow([
            MODEL["name"], dataset["name"], tf_count,
            *(format(value, ".9g") for value in mean_vector.tolist()),
        ])
    return output


def main():
    spaces = {key: load_spaces(dataset) for key, dataset in DATASETS.items()}

    cluster_rows, pair_rows = [], []
    for space_index, space in enumerate(("pooled", "raw")):
        members = {key: value[space_index] for key, value in spaces.items()}
        means = {key: value.mean(dim=0) for key, value in members.items()}

        print(f"\n{space.upper()} space")
        for key, dataset in DATASETS.items():
            vectors = members[key]
            rms = (vectors - means[key]).pow(2).sum(dim=1).mean().sqrt().item()
            cluster_rows.append({
                "model": MODEL["name"],
                "space": space,
                "organism": dataset["name"],
                "members": vectors.shape[0],
                "mean_vector_norm": f"{means[key].norm().item():.6f}",
                "rms_in_cluster": f"{rms:.6f}",
                "member_norm_min": f"{vectors.norm(dim=1).min().item():.6f}",
                "member_norm_max": f"{vectors.norm(dim=1).max().item():.6f}",
            })
            print(f"  {dataset['name']:<20} n={vectors.shape[0]:>6}  "
                  f"norm {means[key].norm().item():.4f}  RMS {rms:.4f}")
            if space == "pooled":
                save_mean_vector(dataset, means[key], vectors.shape[0])

        for left, right in combinations(DATASETS, 2):
            cosine = torch.nn.functional.cosine_similarity(
                means[left], means[right], dim=0
            ).item()
            distance = (means[left] - means[right]).norm().item()
            # ||a-b|| can never exceed ||a||+||b||; that needs cosine = -1.
            bound = (means[left].norm() + means[right].norm()).item()
            pair_rows.append({
                "model": MODEL["name"],
                "space": space,
                "organism_a": DATASETS[left]["name"],
                "organism_b": DATASETS[right]["name"],
                "cosine_similarity": f"{cosine:.6f}",
                "euclidean_distance": f"{distance:.6f}",
                "euclidean_max_possible": f"{bound:.6f}",
                "distance_as_fraction_of_max": f"{distance / bound:.6f}",
            })
            print(f"    {DATASETS[left]['name']} vs {DATASETS[right]['name']}: "
                  f"cosine {cosine:.4f}  euclid {distance:.4f} "
                  f"(max possible {bound:.4f})")

    results = PROJECT_DIR / "results"
    results.mkdir(parents=True, exist_ok=True)
    clusters_csv = results / f"{MODEL['name']}_cluster_statistics.csv"
    pairs_csv = results / f"{MODEL['name']}_cluster_pairwise.csv"
    for path, rows in ((clusters_csv, cluster_rows), (pairs_csv, pair_rows)):
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved: {clusters_csv}")
    print(f"Saved: {pairs_csv}")


if __name__ == "__main__":
    main()
