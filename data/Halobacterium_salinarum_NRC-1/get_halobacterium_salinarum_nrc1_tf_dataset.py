"""Build the H. salinarum NRC-1 TF set: the union of MiST4 and UniProtKB.

Membership is every protein in either source:

  - MiST4: all genes on genome GCF_000006805.1 whose output domain MiST labels
    "DNA binding" (get_mist_signal_genes.py), with no further filtering.
  - UniProtKB: `(organism_id:64091) AND (go:0003700) AND (fragment:false)`,
    which also matches GO:0003700's child terms.

Each MiST gene is linked to UniProt by its locus tag (MiST's old tag VNG6478H
written as UniProt's VNG_6478H) or, failing that, by its RefSeq protein
accession, which UniProt cross-references. Linked genes take their sequence
and annotation from UniProt; genes with no UniProt entry take the RefSeq
protein sequence served by MiST4. Identical plasmid copies collapse to one
protein row that records every locus tag.
"""

import csv
import re
from io import StringIO
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
MIST_URL = "https://mib-jouline-db.asc.ohio-state.edu/v1"
QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
GO_ID = "GO:0003700"
GO_QUERY = "(organism_id:{organism}) AND (go:0003700) AND (fragment:false)"
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
    "xref_refseq",
    "fragment",
    "length",
    "sequence",
]
DATA_DIR = Path(__file__).resolve().parent / "data"
MIST_INPUT = DATA_DIR / "halobacterium_salinarum_nrc1_mist_dna_binding_genes.csv"
OUTPUT = DATA_DIR / "halobacterium_salinarum_nrc1_tf_dataset.csv"


def api_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def uniprot_locus(old_locus):
    """MiST old locus VNG6478H -> UniProt ordered locus name VNG_6478H."""
    match = re.fullmatch(r"VNG(\d.*)", old_locus or "")
    return f"VNG_{match.group(1)}" if match else None


def split_ids(value):
    return [item for item in re.split(r"[;\s]+", value or "") if item]


def read_mist():
    with MIST_INPUT.open(newline="") as source:
        rows = list(csv.DictReader(source))
    organism_ids = {row["taxonomy_id"] for row in rows}
    if len(organism_ids) != 1:
        raise ValueError(f"MiST file spans several organism IDs: {organism_ids}")
    return rows, organism_ids.pop()


def tf_go_terms(session):
    response = session.get(
        f"{QUICKGO_URL}/{GO_ID}/descendants",
        params={"relations": "is_a,part_of"},
        timeout=90,
    )
    response.raise_for_status()
    return {GO_ID} | set(response.json()["results"][0].get("descendants", []))


def uniprot_tsv(session, query, fields):
    response = session.get(
        UNIPROT_URL,
        params={"query": query, "format": "tsv", "fields": ",".join(fields)},
        timeout=240,
    )
    response.raise_for_status()
    rows = list(csv.DictReader(StringIO(response.text), delimiter="\t"))
    release = {
        "uniprot_release": response.headers.get("X-UniProt-Release", ""),
        "uniprot_release_date": response.headers.get("X-UniProt-Release-Date", ""),
    }
    return rows, release


def mist_sequence(session, aseq_id):
    response = session.get(f"{MIST_URL}/aseqs/{aseq_id}", timeout=60)
    response.raise_for_status()
    record = response.json()
    sequence = record["sequence"]
    if len(sequence) != record["length"]:
        raise ValueError(f"MiST sequence {aseq_id}: length mismatch")
    return sequence


def link_mist_gene(mist, by_locus, by_refseq):
    tag = uniprot_locus(mist["old_locus_tag"])
    accessions = by_locus.get(tag, set()) if tag else set()
    if len(accessions) == 1:
        return next(iter(accessions)), "locus_tag"
    refseq = mist["refseq_protein"].split(".")[0]
    accessions = by_refseq.get(refseq, set()) if refseq else set()
    if len(accessions) == 1:
        return next(iter(accessions)), "refseq_protein"
    if len(accessions) > 1:
        raise ValueError(f"{mist['locus_tag']}: RefSeq {refseq} matches {sorted(accessions)}")
    return None, "mist_only"


MIST_ID_FIELDS = {
    "mist_locus_tag": "locus_tag",
    "mist_old_locus_tag": "old_locus_tag",
    "refseq_protein": "refseq_protein",
}


def append_mist_ids(record, mist):
    """Record another MiST locus that resolved to the same protein."""
    for field, column in MIST_ID_FIELDS.items():
        value = mist[column]
        present = [item for item in record[field].split("|") if item]
        if value and value not in present:
            record[field] = "|".join(present + [value])


def uniprot_fields(row, tf_terms, organism_id):
    if row["Organism (ID)"] != organism_id:
        raise ValueError(f"{row['Entry']}: organism {row['Organism (ID)']}")
    if int(row["Length"]) != len(row["Sequence"]):
        raise ValueError(f"{row['Entry']}: sequence-length mismatch")
    go_terms = [term.strip() for term in row["Gene Ontology IDs"].split(";") if term.strip()]
    locus_tags = split_ids(row["Gene Names (ordered locus)"])
    primary_gene = row["Gene Names (primary)"].strip()
    return {
        "gene": primary_gene or (locus_tags[0] if locus_tags else row["Entry"]),
        "locus_tag": "|".join(locus_tags),
        "uniprot_accession": row["Entry"],
        "uniprot_entry_name": row["Entry Name"],
        "uniprot_gene_name": primary_gene,
        "protein_name": row["Protein names"],
        "reviewed": str(row["Reviewed"] == "reviewed").lower(),
        "organism_name": row["Organism"],
        "organism_id": row["Organism (ID)"],
        "go_terms": "|".join(go_terms),
        "tf_go_terms": "|".join(sorted(set(go_terms) & tf_terms)),
        "fragment": str(row["Fragment"] == "fragment").lower(),
        "sequence_length": len(row["Sequence"]),
        "sequence": row["Sequence"],
    }


def build(session):
    mist_rows, organism_id = read_mist()
    tf_terms = tf_go_terms(session)
    proteome, release = uniprot_tsv(session, f"(organism_id:{organism_id})", FIELDS)
    go_rows, _ = uniprot_tsv(session, GO_QUERY.format(organism=organism_id), ["accession"])
    go_accessions = {row["Entry"] for row in go_rows}

    by_accession = {row["Entry"]: row for row in proteome}
    by_locus, by_refseq = {}, {}
    for row in proteome:
        for tag in split_ids(row["Gene Names (ordered locus)"]):
            by_locus.setdefault(tag, set()).add(row["Entry"])
        for refseq in split_ids(row["RefSeq"]):
            by_refseq.setdefault(refseq.split(".")[0], set()).add(row["Entry"])

    proteins = {}
    for mist in mist_rows:
        accession, link = link_mist_gene(mist, by_locus, by_refseq)
        key = accession or f"mist:{mist['mist_aseq_id']}"
        if key not in proteins:
            if accession:
                record = uniprot_fields(by_accession[accession], tf_terms, organism_id)
                record["sequence_source"] = "uniprot"
            else:
                sequence = mist_sequence(session, mist["mist_aseq_id"])
                record = {
                    "gene": mist["old_locus_tag"] or mist["locus_tag"],
                    "locus_tag": mist["old_locus_tag"],
                    "uniprot_accession": "",
                    "uniprot_entry_name": "",
                    "uniprot_gene_name": "",
                    "protein_name": mist["mist_product"],
                    "reviewed": "",
                    "organism_name": mist["genome_name"],
                    "organism_id": organism_id,
                    "go_terms": "",
                    "tf_go_terms": "",
                    "fragment": "",
                    "sequence_length": len(sequence),
                    "sequence": sequence,
                    "sequence_source": "mist4_refseq",
                }
            record.update({
                "in_mist": "true",
                "in_uniprot_go": str(accession in go_accessions).lower(),
                "mist_link": link,
                "mist_locus_tag": mist["locus_tag"],
                "mist_old_locus_tag": mist["old_locus_tag"],
                "refseq_protein": mist["refseq_protein"],
                "mist_product": mist["mist_product"],
                "mist_ranks": mist["mist_ranks"],
                "mist_dna_binding_domains": mist["mist_dna_binding_domains"],
            })
            proteins[key] = record
        else:
            append_mist_ids(proteins[key], mist)

    # A MiST-only protein can still be identical to a UniProt protein that
    # neither its locus tag nor its RefSeq accession links to; merge those.
    by_sequence = {
        record["sequence"]: key
        for key, record in proteins.items()
        if record["sequence_source"] == "uniprot"
    }
    for key in [k for k, r in proteins.items() if r["sequence_source"] == "mist4_refseq"]:
        match = by_sequence.get(proteins[key]["sequence"])
        if match:
            record = proteins.pop(key)
            proteins[match]["mist_link"] = proteins[match]["mist_link"] or "identical_sequence"
            append_mist_ids(proteins[match], {
                "locus_tag": record["mist_locus_tag"],
                "old_locus_tag": record["mist_old_locus_tag"],
                "refseq_protein": record["refseq_protein"],
            })

    for accession in sorted(go_accessions - set(proteins)):
        record = uniprot_fields(by_accession[accession], tf_terms, organism_id)
        record.update({
            "sequence_source": "uniprot",
            "in_mist": "false",
            "in_uniprot_go": "true",
            "mist_link": "",
            "mist_locus_tag": "",
            "mist_old_locus_tag": "",
            "refseq_protein": "",
            "mist_product": "",
            "mist_ranks": "",
            "mist_dna_binding_domains": "",
        })
        proteins[accession] = record

    order = [
        "gene", "locus_tag", "uniprot_accession", "uniprot_entry_name",
        "uniprot_gene_name", "protein_name", "reviewed", "organism_name",
        "organism_id", "go_terms", "tf_go_terms", "in_mist", "in_uniprot_go",
        "mist_link", "mist_locus_tag", "mist_old_locus_tag", "refseq_protein",
        "mist_product", "mist_ranks", "mist_dna_binding_domains", "fragment",
        "sequence_source", "sequence_length", "sequence",
    ]
    cleaned = []
    for record in proteins.values():
        row = {field: record[field] for field in order}
        row.update(release)
        row["selection"] = "union: MiST4 output/DNA binding OR UniProt GO:0003700"
        cleaned.append(row)
    sequences = [row["sequence"] for row in cleaned]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Two protein rows share an identical sequence")
    return sorted(cleaned, key=lambda row: (row["locus_tag"] or "~", row["uniprot_accession"])), mist_rows, go_accessions, release


def main():
    session = api_session()
    cleaned, mist_rows, go_accessions, release = build(session)

    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    both = sum(row["in_mist"] == "true" and row["in_uniprot_go"] == "true" for row in cleaned)
    mist_only = sum(row["in_mist"] == "true" and row["in_uniprot_go"] == "false" for row in cleaned)
    uniprot_only = sum(row["in_mist"] == "false" for row in cleaned)
    links = {}
    for row in cleaned:
        if row["mist_link"]:
            links[row["mist_link"]] = links.get(row["mist_link"], 0) + 1
    print(f"UniProt release: {release['uniprot_release']}")
    print(f"MiST4 DNA-binding genes: {len(mist_rows)}   UniProt GO:0003700 TFs: {len(go_accessions)}")
    print(f"MiST proteins linked by: {links}")
    print(f"Union: {len(cleaned)} distinct proteins")
    print(f"  in both: {both}   MiST4 only: {mist_only}   UniProt only: {uniprot_only}")
    print(f"  sequences from MiST4 (no UniProt entry): {sum(row['sequence_source'] == 'mist4_refseq' for row in cleaned)}")
    print(f"Saved dataset: {OUTPUT}")


if __name__ == "__main__":
    main()
