"""Download and clean the E. coli K-12 transcription-factor catalog."""

import csv
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://regulondb.ccg.unam.mx/graphql"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_TF_DATASET = DATA_DIR / "regulon_tf_dataset_raw.tsv"
CLEAN_DATASET = DATA_DIR / "regulon_tf_dataset_clean.csv"

DOWNLOAD_QUERY = """
query DownloadTfDataset {
  tfSet: getDataOfFile(fileName: "TFSet") {
    fileName
    version
    rdbVersion
    content
  }
  geneProducts: getDataOfFile(fileName: "GeneProductAllIdentifiersSet") {
    fileName
    version
    rdbVersion
    content
  }
}
"""


def api_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=None,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def download_sources():
    response = api_session().post(
        API_URL,
        json={"query": DOWNLOAD_QUERY},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(f"RegulonDB GraphQL error: {payload['errors']}")

    return payload["data"]["tfSet"], payload["data"]["geneProducts"]


def parse_tsv(content):
    reader = csv.DictReader(StringIO(content), delimiter="\t")
    return [
        {
            key.replace("\xa0", "").strip(): value.strip()
            for key, value in row.items()
        }
        for row in reader
    ]


def split_pipe(value):
    return [item.strip() for item in value.split("|") if item.strip()]


def identifiers(value, database):
    return re.findall(rf"\[{database}:([^\]]+)\]", value)


def build_uniprot_lookup(gene_product_rows):
    accessions_by_bnumber = defaultdict(set)

    for row in gene_product_rows:
        bnumbers = identifiers(row["7)otherDbsGeneIds"], "REFSEQ")
        accessions = identifiers(row["11)otherDbsProductsIds"], "UNIPROT")

        for bnumber in bnumbers:
            accessions_by_bnumber[bnumber].update(accessions)

    lookup = {}

    for bnumber, accessions in accessions_by_bnumber.items():
        canonical = [
            accession
            for accession in accessions
            if "-" not in accession
        ]

        if len(accessions) == 1:
            lookup[bnumber] = next(iter(accessions))
        elif len(canonical) == 1:
            lookup[bnumber] = canonical[0]

    return lookup


def clean_tf_dataset(tf_rows, uniprot_by_bnumber, regulondb_version):
    proteins = defaultdict(
        lambda: {
            "gene": "",
            "bnumber": "",
            "uniprot_accession": "",
            "tf_entities": [],
            "tf_regulondb_ids": [],
        }
    )

    for row in tf_rows:
        genes = split_pipe(row["4)geneCodingForTF"])
        bnumbers = split_pipe(row["5)geneBnumberCodingForTF"])

        if len(genes) != len(bnumbers):
            raise ValueError(
                f"{row['2)name']}: {len(genes)} genes but "
                f"{len(bnumbers)} b-numbers"
            )

        for gene, bnumber in zip(genes, bnumbers):
            protein = proteins[bnumber]
            protein["gene"] = gene
            protein["bnumber"] = bnumber
            protein["uniprot_accession"] = uniprot_by_bnumber.get(
                bnumber,
                "",
            )

            for source, destination in (
                (row["2)name"], "tf_entities"),
                (row["1)id"], "tf_regulondb_ids"),
            ):
                if source not in protein[destination]:
                    protein[destination].append(source)

    cleaned = []
    for protein in proteins.values():
        cleaned.append({
            "gene": protein["gene"],
            "bnumber": protein["bnumber"],
            "uniprot_accession": protein["uniprot_accession"],
            "tf_entities": "|".join(protein["tf_entities"]),
            "tf_regulondb_ids": "|".join(protein["tf_regulondb_ids"]),
            "num_tf_entities": len(protein["tf_entities"]),
            "regulondb_version": regulondb_version,
        })

    return sorted(cleaned, key=lambda row: row["bnumber"])


def write_csv(rows):
    with CLEAN_DATASET.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tf_source, gene_product_source = download_sources()

    RAW_TF_DATASET.write_text(tf_source["content"])

    tf_rows = parse_tsv(tf_source["content"])
    gene_product_rows = parse_tsv(gene_product_source["content"])
    uniprot_by_bnumber = build_uniprot_lookup(gene_product_rows)
    cleaned = clean_tf_dataset(
        tf_rows,
        uniprot_by_bnumber,
        tf_source["rdbVersion"],
    )
    write_csv(cleaned)

    missing = [row for row in cleaned if not row["uniprot_accession"]]
    print(f"RegulonDB version: {tf_source['rdbVersion']}")
    print(f"Raw TF entities: {len(tf_rows)}")
    print(f"Unique TF proteins: {len(cleaned)}")
    print(f"UniProt accessions: {len(cleaned) - len(missing)}/{len(cleaned)}")
    print(f"Saved raw TF data: {RAW_TF_DATASET}")
    print(f"Saved clean dataset: {CLEAN_DATASET}")

    if missing:
        names = ", ".join(row["gene"] for row in missing)
        raise RuntimeError(f"TFs without a UniProt accession: {names}")


if __name__ == "__main__":
    main()
