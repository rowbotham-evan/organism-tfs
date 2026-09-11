"""Build the H. volcanii DS2 TF set by domain reconstruction from UniProtKB.

Haloferax volcanii has no equivalent of the Network Portal catalog, so
membership rests on the domain route alone:

  1. UniProtKB proteome for taxon 309800, keeping proteins that carry
     GO:0003700 (DNA-binding transcription factor activity), a Pfam
     helix-turn-helix family, or a basal transcription factor domain
     (TBP, TFIIB).
  2. Drop entries whose UniProt name is affirmatively non-regulatory --
     enzymes, helicases, translation and ribosome machinery. The winged-helix
     fold is shared by many nucleotide-binding enzymes.

Sequences are always the current UniProtKB canonical ones.
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


API_URL = "https://rest.uniprot.org/uniprotkb/stream"
TAXON_ID = "309800"
PROTEOME_QUERY = f"(organism_id:{TAXON_ID})"
FIELDS = [
    "accession", "id", "reviewed", "protein_name", "gene_primary", "gene_oln",
    "organism_name", "organism_id", "go_id", "xref_pfam", "fragment",
    "length", "sequence",
]
TF_GO_ID = "GO:0003700"

# Pfam helix-turn-helix and basal-factor families occurring in the H. volcanii
# proteome, from InterPro's Pfam entries matching helix-turn-helix / winged
# helix / HTH, plus TBP (PF00352) and TFIIB (PF00382, PF08271).
TF_PFAM_FAMILIES = frozenset((
    "PF00352", "PF00382", "PF01022", "PF01381", "PF01638", "PF01866",
    "PF01902", "PF03444", "PF04967", "PF08220", "PF08271", "PF08279",
    "PF09079", "PF09339", "PF12840", "PF13404", "PF13412", "PF13551",
    "PF13560", "PF13565", "PF13592", "PF15915", "PF19306", "PF19575",
    "PF20575", "PF22451", "PF22665", "PF22982", "PF23336", "PF23445",
    "PF24250", "PF24266", "PF24270", "PF25947", "PF26271", "PF26462",
    "PF26491", "PF28410", "PF28603", "PF29290",
))

NON_REGULATORY = re.compile(
    r"dehydrogenase|synthase|synthetase|kinase|transferase|reductase|ligase|"
    r"hydrolase|isomerase|lyase|mutase|peptidase|protease|nuclease|polymerase|"
    r"helicase|topoisomerase|ATPase|oxidase|carboxylase|phosphatase|"
    r"ribosomal|chaperone|nascent polypeptide|transporter|permease|"
    r"antitermination|elongation factor|translation|tRNA|rRNA|"
    r"cell division|flagell|gas vesicle|transposase|integrase|recombinase",
    re.IGNORECASE,
)
REGULATORY = re.compile(
    r"transcription(al)? (regulator|activator|repressor)|"
    r"DNA-binding transcription",
    re.IGNORECASE,
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "haloferax_volcanii_ds2_tf_dataset.csv"
)


def api_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def download_proteome():
    response = api_session().get(
        API_URL,
        params={"query": PROTEOME_QUERY, "format": "tsv",
                 "fields": ",".join(FIELDS)},
        timeout=240,
    )
    response.raise_for_status()
    rows = list(csv.DictReader(StringIO(response.text), delimiter="\t"))
    return rows, {
        "uniprot_release": response.headers.get("X-UniProt-Release", ""),
        "uniprot_release_date": response.headers.get("X-UniProt-Release-Date", ""),
    }


def select_tfs(proteome):
    selected, dropped = {}, []
    for row in proteome:
        pfam = {p for p in (row["Pfam"] or "").split(";") if p.strip()}
        reasons = []
        if TF_GO_ID in (row["Gene Ontology IDs"] or ""):
            reasons.append("go_0003700")
        if pfam & TF_PFAM_FAMILIES:
            reasons.append("pfam_tf_domain")
        if not reasons:
            continue

        name = row["Protein names"] or ""
        if NON_REGULATORY.search(name) and not REGULATORY.search(name):
            dropped.append((row["Gene Names (ordered locus)"], name))
            continue
        selected[row["Entry"]] = (row, reasons)
    return selected, dropped


def clean_dataset(selected, release):
    cleaned = []
    for accession, (row, reasons) in selected.items():
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
            "gene": primary_gene or (locus_tags[0] if locus_tags else accession),
            "locus_tag": "|".join(locus_tags),
            "uniprot_accession": accession,
            "uniprot_entry_name": row["Entry Name"],
            "protein_name": row["Protein names"],
            "reviewed": str(row["Reviewed"] == "reviewed").lower(),
            "organism_name": row["Organism"],
            "organism_id": row["Organism (ID)"],
            "go_terms": "|".join(t.strip() for t in
                                 row["Gene Ontology IDs"].split(";") if t.strip()),
            "pfam": "|".join(sorted(p for p in (row["Pfam"] or "").split(";") if p.strip())),
            "sequence_length": sequence_length,
            "sequence": sequence,
            **release,
            "evidence_sources": "|".join(sorted(reasons)),
            "evidence_source_count": len(reasons),
        })

    if not cleaned:
        raise ValueError("No H. volcanii TFs selected")
    return sorted(cleaned, key=lambda row: (row["locus_tag"] or "~",
                                            row["uniprot_accession"]))


def main():
    proteome, release = download_proteome()
    selected, dropped = select_tfs(proteome)
    cleaned = clean_dataset(selected, release)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    counts = {}
    for row in cleaned:
        for reason in row["evidence_sources"].split("|"):
            counts[reason] = counts.get(reason, 0) + 1
    print(f"UniProt release: {release['uniprot_release']}")
    print(f"Proteome entries scanned: {len(proteome)}")
    print(f"Dropped as non-regulatory: {len(dropped)}")
    print(f"Selected TFs: {len(cleaned)}")
    for reason, count in sorted(counts.items()):
        print(f"  {reason}: {count}")
    print(f"  supported by both routes: "
          f"{sum(row['evidence_source_count'] > 1 for row in cleaned)}")
    print(f"Saved dataset: {OUTPUT}")


if __name__ == "__main__":
    main()
