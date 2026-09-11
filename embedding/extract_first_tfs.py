"""Export the first raw TF embeddings to residue-level CSV files."""

import argparse
import csv
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_CONFIGS = {
    "300m": {"name": "ESMC-300M", "slug": "esmc-300M", "width": 960},
    "600m": {"name": "ESMC-600M", "slug": "esmc-600M", "width": 1152},
}
DATASET_CONFIGS = {
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
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_CONFIGS, "all"), default="all")
    parser.add_argument(
        "--dataset",
        choices=(*DATASET_CONFIGS, "all"),
        default="all",
    )
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def extract(model_key, dataset_key, count):
    config = MODEL_CONFIGS[model_key]
    dataset = DATASET_CONFIGS[dataset_key]
    model_dir = PROJECT_DIR / "embedding" / config["slug"]
    source = model_dir / f"{config['slug']}_{dataset['source_suffix']}"
    records = torch.load(source, map_location="cpu", weights_only=False)[:count]

    if len(records) != count:
        raise ValueError(f"Requested {count} records, but {source} has {len(records)}")
    for record in records:
        expected = (len(record["sequence"]), config["width"])
        if tuple(record["embedding"].shape) != expected:
            raise ValueError(
                f"{record['gene']}: expected {expected}, "
                f"got {tuple(record['embedding'].shape)}"
            )

    output = (
        model_dir
        / f"{config['slug']}_{dataset['output_slug']}_first_{count}_raw_embeddings.csv"
    )
    header = [
        "model",
        "organism",
        "gene",
        "dataset_id",
        "uniprot_accession",
        "residue_index",
        "amino_acid",
        *(f"embedding_{index}" for index in range(1, config["width"] + 1)),
    ]
    with output.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(header)
        for record in records:
            metadata = [
                config["name"],
                dataset["name"],
                record["gene"],
                record.get("bnumber", record.get("locus_tag", "")),
                record["uniprot_accession"],
            ]
            for residue_index, (amino_acid, vector) in enumerate(
                zip(record["sequence"], record["embedding"], strict=True),
                start=1,
            ):
                writer.writerow([
                    *metadata,
                    residue_index,
                    amino_acid,
                    *(format(value, ".9g") for value in vector.tolist()),
                ])
    return records, output


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    model_keys = MODEL_CONFIGS if args.model == "all" else (args.model,)
    dataset_keys = (
        DATASET_CONFIGS if args.dataset == "all" else (args.dataset,)
    )
    for model_key in model_keys:
        for dataset_key in dataset_keys:
            records, output = extract(model_key, dataset_key, args.count)
            print(f"Saved: {output}")
            for record in records:
                print(
                    f"  {record['gene']} ({record['uniprot_accession']}): "
                    f"{tuple(record['embedding'].shape)}"
                )


if __name__ == "__main__":
    main()
