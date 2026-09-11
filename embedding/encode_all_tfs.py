"""Generate residue-level ESMC embeddings for every cataloged TF."""

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_CONFIGS = {
    "600m": {
        "name": "ESMC-600M",
        "slug": "esmc-600M",
        "path": PROJECT_DIR / "models" / "esmc-600M",
        "width": 1152,
    },
}
DATASET_CONFIGS = {
    "ecoli": {
        "path": (
            PROJECT_DIR
            / "data"
            / "Escherichia_coli_K-12"
            / "data"
            / "regulon_tf_dataset_with_aa_sequences.csv"
        ),
        "output_suffix": "all_raw_embeddings.pt",
        "metadata": (
            "gene",
            "bnumber",
            "uniprot_accession",
            "tf_entities",
        ),
    },
    "nrc1": {
        "path": (
            PROJECT_DIR
            / "data"
            / "Halobacterium_salinarum_NRC-1"
            / "data"
            / "halobacterium_salinarum_nrc1_tf_dataset.csv"
        ),
        "output_suffix": "halobacterium_salinarum_nrc1_tf_embeddings.pt",
        "metadata": (
            "gene",
            "locus_tag",
            "uniprot_accession",
            "uniprot_entry_name",
            "protein_name",
            "reviewed",
            "organism_id",
            "go_terms",
        ),
    },
    "hvolc": {
        "path": (
            PROJECT_DIR
            / "data"
            / "Haloferax_volcanii_DS2"
            / "data"
            / "haloferax_volcanii_ds2_tf_dataset.csv"
        ),
        "output_suffix": "haloferax_volcanii_ds2_tf_embeddings.pt",
        "metadata": (
            "gene",
            "locus_tag",
            "uniprot_accession",
            "uniprot_entry_name",
            "protein_name",
            "reviewed",
            "organism_id",
            "go_terms",
        ),
    },
    "pao1": {
        "path": (
            PROJECT_DIR
            / "data"
            / "Pseudomonas_aeruginosa_PAO1"
            / "data"
            / "pseudomonas_aeruginosa_pao1_tf_dataset.csv"
        ),
        "output_suffix": "pseudomonas_aeruginosa_pao1_tf_embeddings.pt",
        "metadata": (
            "gene",
            "locus_tag",
            "uniprot_accession",
            "uniprot_entry_name",
            "protein_name",
            "reviewed",
            "organism_id",
            "go_terms",
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_CONFIGS,
        default="600m",
        help="ESMC checkpoint to use (default: 600m)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIGS,
        default="ecoli",
        help="TF dataset to embed (default: ecoli)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    dataset_config = DATASET_CONFIGS[args.dataset]
    output = (
        PROJECT_DIR
        / "embedding"
        / model_config["slug"]
        / f"{model_config['slug']}_{dataset_config['output_suffix']}"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not model_config["path"].is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_config['path']}"
        )

    print(f"Using {model_config['name']} on {device}")
    print(f"Dataset: {args.dataset}")
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    model = AutoModel.from_pretrained(model_config["path"]).to(device).eval()

    with dataset_config["path"].open(newline="") as source:
        proteins = list(csv.DictReader(source))

    records = []
    for index, protein in enumerate(proteins, start=1):
        sequence = protein["sequence"]
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_tokens = inputs.pop("special_tokens_mask").to(device)
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state

        residue_mask = inputs["attention_mask"].bool() & ~special_tokens.bool()
        embedding = hidden[residue_mask].cpu()
        expected_shape = (len(sequence), model_config["width"])

        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"{protein['gene']}: expected {expected_shape}, "
                f"got {tuple(embedding.shape)}"
            )

        record = {
            key: protein[key]
            for key in dataset_config["metadata"]
        }
        records.append({
            **record,
            "sequence": sequence,
            "source_dataset": args.dataset,
            "esmc_model": model_config["name"],
            "embedding": embedding,
        })
        print(
            f"[{index}/{len(proteins)}] {protein['gene']}: "
            f"{len(sequence)} aa -> {tuple(embedding.shape)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output)
    print(f"Saved {len(records)} embeddings: {output}")


if __name__ == "__main__":
    main()
