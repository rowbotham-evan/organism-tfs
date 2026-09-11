"""Build the P. aeruginosa PAO1 TF set: MiST4 genes, verified and sequenced in UniProtKB.

1. get_mist_signal_genes.py collects the genes on MiST4's page for this genome
   with kind=output and function=DNA binding, keeping each gene's locus tag and
   the genome's organism (taxonomy) ID.
2. UniProtKB is queried for the entries matching all of: the MiST organism ID,
   the MiST locus tag, and GO:0003700 (DNA-binding transcription factor
   activity). UniProt's go: search also matches the term's child terms, such as
   GO:0001216 (DNA-binding transcription activator activity).
3. Every returned entry is re-checked before its sequence is kept: organism ID,
   exact locus tag, a TF GO term (GO:0003700 or a child term from QuickGO), a
   complete sequence, and exactly one entry per locus tag.

MiST4 holds no GO annotations; GO:0003700 is the criterion this pipeline adds.
"""

import csv
from io import StringIO
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
GO_ID = "GO:0003700"
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


def read_mist():
    with MIST_INPUT.open(newline="") as source:
        rows = list(csv.DictReader(source))
    organism_ids = {row["taxonomy_id"] for row in rows}
    if len(organism_ids) != 1:
        raise ValueError(f"MiST file spans several organism IDs: {organism_ids}")
    return rows, organism_ids.pop()


def tf_go_terms(session):
    """GO:0003700 plus every term below it, the set UniProt's go: search matches."""
    response = session.get(
        f"{QUICKGO_URL}/{GO_ID}/descendants",
        params={"relations": "is_a,part_of"},
        timeout=90,
    )
    response.raise_for_status()
    children = set(response.json()["results"][0].get("descendants", []))
    return {GO_ID} | children


def query_uniprot(session, organism_id, loci):
    """Ask UniProt for entries matching organism ID AND locus tag AND GO:0003700."""
    matches = {locus: [] for locus in loci}
    release = {}
    for start in range(0, len(loci), BATCH_SIZE):
        batch = loci[start:start + BATCH_SIZE]
        response = session.get(
            UNIPROT_URL,
            params={
                "query": (
                    f"(organism_id:{organism_id}) AND "
                    f"(gene:({' OR '.join(batch)})) AND "
                    f"(go:{GO_ID[3:]})"
                ),
                "format": "tsv",
                "fields": ",".join(FIELDS),
            },
            timeout=180,
        )
        response.raise_for_status()
        release = {
            "uniprot_release": response.headers.get("X-UniProt-Release", ""),
            "uniprot_release_date": response.headers.get(
                "X-UniProt-Release-Date", ""
            ),
        }
        for row in csv.DictReader(StringIO(response.text), delimiter="\t"):
            # Only an exact locus-tag match counts; the gene: search also
            # matches names that merely contain the tag.
            for tag in row["Gene Names (ordered locus)"].split():
                if tag in matches:
                    matches[tag].append(row)
    return matches, release


def verify(mist_rows, matches, organism_id, tf_terms, release):
    cleaned, not_returned = [], []
    for mist in mist_rows:
        locus = mist["locus_tag"]
        entries = matches[locus]
        if not entries:
            not_returned.append(locus)
            continue
        if len(entries) > 1:
            raise ValueError(
                f"{locus} matches several UniProt entries: "
                f"{[entry['Entry'] for entry in entries]}"
            )
        row = entries[0]
        accession = row["Entry"]
        go_terms = [
            term.strip()
            for term in row["Gene Ontology IDs"].split(";")
            if term.strip()
        ]
        matched_tf_terms = sorted(set(go_terms) & tf_terms)
        sequence = row["Sequence"]
        sequence_length = int(row["Length"])

        if row["Organism (ID)"] != organism_id:
            raise ValueError(f"{accession}: organism {row['Organism (ID)']}, expected {organism_id}")
        if not matched_tf_terms:
            raise ValueError(f"{accession}: no {GO_ID} or child term")
        if row["Fragment"] == "fragment":
            raise ValueError(f"{accession}: fragment")
        if sequence_length != len(sequence):
            raise ValueError(f"{accession}: sequence-length mismatch")

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
            "go_terms": "|".join(go_terms),
            "tf_go_terms": "|".join(matched_tf_terms),
            "mist_locus_tag": locus,
            "mist_product": mist["mist_product"],
            "mist_ranks": mist["mist_ranks"],
            "mist_dna_binding_domains": mist["mist_dna_binding_domains"],
            "sequence_length": sequence_length,
            "sequence": sequence,
            **release,
            "selection": "MiST4 output/DNA binding; UniProt organism_id AND locus tag AND GO:0003700",
            "uniprot_query": (
                f"(organism_id:{organism_id}) AND (gene:<MiST locus tag>) AND (go:{GO_ID[3:]})"
            ),
        })

    accessions = [row["uniprot_accession"] for row in cleaned]
    if len(accessions) != len(set(accessions)):
        raise ValueError("Two MiST loci resolved to the same UniProt accession")
    if not cleaned:
        raise ValueError("No MiST4 genes verified in UniProt")
    return sorted(cleaned, key=lambda row: row["mist_locus_tag"]), not_returned


def main():
    mist_rows, organism_id = read_mist()
    loci = [row["locus_tag"] for row in mist_rows]

    session = api_session()
    tf_terms = tf_go_terms(session)
    matches, release = query_uniprot(session, organism_id, loci)
    cleaned, not_returned = verify(mist_rows, matches, organism_id, tf_terms, release)

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    literal = sum(GO_ID in row["tf_go_terms"].split("|") for row in cleaned)
    print(f"UniProt release: {release['uniprot_release']}")
    print(f"MiST4 DNA-binding genes: {len(mist_rows)} (organism ID {organism_id})")
    print(f"TF GO terms accepted: {GO_ID} + {len(tf_terms) - 1} child terms")
    print(f"MiST loci with no verified UniProt entry: {len(not_returned)}")
    print(f"Verified TFs saved: {len(cleaned)}")
    print(f"  with {GO_ID} itself: {literal}   with only a child term: {len(cleaned) - literal}")
    print(f"Saved dataset: {OUTPUT}")


if __name__ == "__main__":
    main()
