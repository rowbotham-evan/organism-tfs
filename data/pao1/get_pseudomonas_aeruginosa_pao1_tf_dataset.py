"""Build the P. aeruginosa PAO1 TF set from MiST4, with UniProtKB sequences.

MiST4 is the primary source: get_mist_signal_genes.py lists the genes MiST
classifies as having a DNA-binding output domain. Each MiST locus tag is then
looked up in UniProtKB, the way RegulonDB b-numbers are for E. coli, and a TF
is kept only if UniProt also annotates it with DNA-binding transcription
factor activity (GO:0003700 or a child term). The dataset is therefore the
intersection of the two databases.
"""

import csv
from io import StringIO
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://rest.uniprot.org/uniprotkb/stream"
TAXON_ID = "208964"
GO_ID = "GO:0003700"
GO_QUERY = f"(organism_id:{TAXON_ID}) AND (go:{GO_ID[3:]}) AND (fragment:false)"
# UniProt allows at most 100 OR terms in one query.
BATCH_SIZE = 100
FIELDS = [
    "accession",
    "id",
    "reviewed",
    "protein_name",
    "gene_primary",
    "gene_oln",
    "organism_name",
    "organism_id",
    "go_id",
    "fragment",
    "length",
    "sequence",
]
DATA_DIR = Path(__file__).resolve().parent / "data"
MIST_INPUT = DATA_DIR / "pseudomonas_aeruginosa_pao1_mist_dna_binding_genes.csv"
OUTPUT = DATA_DIR / "pseudomonas_aeruginosa_pao1_tf_dataset.csv"


def api_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def uniprot(session, params):
    response = session.get(API_URL, params=params, timeout=180)
    response.raise_for_status()
    return response


def release_of(response):
    return {
        "uniprot_release": response.headers.get("X-UniProt-Release", ""),
        "uniprot_release_date": response.headers.get("X-UniProt-Release-Date", ""),
    }


def lookup_loci(session, loci):
    """Map each MiST locus tag to the UniProt entries whose locus names include it."""
    matches = {locus: [] for locus in loci}
    release = {}
    for start in range(0, len(loci), BATCH_SIZE):
        batch = loci[start:start + BATCH_SIZE]
        response = uniprot(session, {
            "query": f"(organism_id:{TAXON_ID}) AND (gene:({' OR '.join(batch)}))",
            "format": "tsv",
            "fields": ",".join(FIELDS),
        })
        release = release_of(response)
        for row in csv.DictReader(StringIO(response.text), delimiter="\t"):
            for tag in row["Gene Names (ordered locus)"].split():
                if tag in matches:
                    matches[tag].append(row)
    return matches, release


def go_tf_accessions(session):
    response = uniprot(session, {"query": GO_QUERY, "format": "list"})
    accessions = set(response.text.split())
    if not accessions:
        raise ValueError("UniProt returned no GO:0003700 proteins")
    return accessions


def build_dataset(mist_rows, matches, go_accessions, release):
    unmatched, not_go, cleaned = [], [], []
    for mist in mist_rows:
        locus = mist["locus_tag"]
        entries = matches[locus]
        if not entries:
            unmatched.append(locus)
            continue
        if len(entries) > 1:
            raise ValueError(
                f"{locus} matches several UniProt entries: "
                f"{[entry['Entry'] for entry in entries]}"
            )
        row = entries[0]
        accession = row["Entry"]
        if accession not in go_accessions:
            not_go.append(locus)
            continue

        sequence = row["Sequence"]
        sequence_length = int(row["Length"])
        if row["Organism (ID)"] != TAXON_ID:
            raise ValueError(f"Unexpected taxon for {accession}")
        if row["Fragment"] == "fragment":
            raise ValueError(f"Fragment returned for {accession}")
        if sequence_length != len(sequence):
            raise ValueError(f"Sequence-length mismatch for {accession}")

        locus_tags = row["Gene Names (ordered locus)"].split()
        primary_gene = row["Gene Names (primary)"].strip()
        cleaned.append({
            "gene": primary_gene or locus,
            "locus_tag": "|".join(locus_tags),
            "uniprot_accession": accession,
            "uniprot_entry_name": row["Entry Name"],
            "uniprot_gene_name": primary_gene,
            "protein_name": row["Protein names"],
            "reviewed": str(row["Reviewed"] == "reviewed").lower(),
            "organism_name": row["Organism"],
            "organism_id": row["Organism (ID)"],
            "go_terms": "|".join(
                term.strip()
                for term in row["Gene Ontology IDs"].split(";")
                if term.strip()
            ),
            "mist_locus_tag": locus,
            "mist_product": mist["mist_product"],
            "mist_ranks": mist["mist_ranks"],
            "mist_dna_binding_domains": mist["mist_dna_binding_domains"],
            "sequence_length": sequence_length,
            "sequence": sequence,
            **release,
            "selection": "MiST4 output/DNA binding AND UniProt GO:0003700",
            "uniprot_go_query": GO_QUERY,
        })

    accessions = [row["uniprot_accession"] for row in cleaned]
    if len(accessions) != len(set(accessions)):
        raise ValueError("Two MiST loci resolved to the same UniProt accession")
    if not cleaned:
        raise ValueError("No PAO1 TFs in both MiST4 and UniProt")
    return sorted(cleaned, key=lambda row: row["mist_locus_tag"]), unmatched, not_go


def main():
    with MIST_INPUT.open(newline="") as source:
        mist_rows = list(csv.DictReader(source))
    loci = [row["locus_tag"] for row in mist_rows]

    session = api_session()
    matches, release = lookup_loci(session, loci)
    go_accessions = go_tf_accessions(session)
    cleaned, unmatched, not_go = build_dataset(
        mist_rows, matches, go_accessions, release
    )

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"UniProt release: {release['uniprot_release']}")
    print(f"MiST4 DNA-binding genes: {len(mist_rows)}")
    print(f"  not in UniProt: {len(unmatched)} {unmatched}")
    print(f"  in UniProt but without GO:0003700: {len(not_go)}")
    print(f"UniProt GO:0003700 proteins: {len(go_accessions)}")
    print(f"Saved TFs in both databases: {len(cleaned)}")
    print(f"Saved dataset: {OUTPUT}")


if __name__ == "__main__":
    main()
