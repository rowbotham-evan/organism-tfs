"""Add canonical UniProt sequences, gene names, and organism to the RegulonDB TF catalog."""

import csv
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT = DATA_DIR / "regulon_tf_dataset_clean.csv"

OUTPUT = DATA_DIR / "regulon_tf_dataset_with_aa_sequences.csv"
API_URL = "https://rest.uniprot.org/uniprotkb"


def api_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_sequence(session, accession):
    response = session.get(
        f"{API_URL}/{accession}",
        params={"format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    record = response.json()
    sequence = record["sequence"]["value"]

    if len(sequence) != record["sequence"]["length"]:
        raise ValueError(f"Sequence-length mismatch for {accession}")

    genes = record.get("genes") or [{}]
    return {
        "uniprot_primary_accession": record["primaryAccession"],
        "uniprot_entry_name": record["uniProtkbId"],
        # UniProt's gene name can differ from RegulonDB's `gene` (e.g. nfeR is yqjI).
        "uniprot_gene_name": genes[0].get("geneName", {}).get("value", ""),
        "organism_name": record["organism"]["scientificName"],
        "organism_id": record["organism"]["taxonId"],
        "sequence_length": len(sequence),
        "sequence": sequence,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT.open(newline="") as source:
        proteins = list(csv.DictReader(source))

    missing = [row["gene"] for row in proteins if not row["uniprot_accession"]]
    if missing:
        raise ValueError(f"Missing UniProt accessions for: {', '.join(missing)}")

    session = api_session()
    enriched = []

    for index, protein in enumerate(proteins, start=1):
        accession = protein["uniprot_accession"]
        sequence_data = fetch_sequence(session, accession)
        enriched.append({**protein, **sequence_data})
        print(
            f"[{index}/{len(proteins)}] {protein['gene']}: "
            f"{accession}, {sequence_data['sequence_length']} aa"
        )

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=enriched[0].keys())
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Saved {len(enriched)} complete sequences: {OUTPUT}")


if __name__ == "__main__":
    main()
