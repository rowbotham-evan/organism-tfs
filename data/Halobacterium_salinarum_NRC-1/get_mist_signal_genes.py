"""Download the MiST4 DNA-binding signal genes for H. salinarum NRC-1.

MiST4 classifies every signal-transduction gene in a genome by its protein
domains. This takes the genes whose output domain MiST labels "DNA binding"
(its category for transcription factors, including sigma factors), the same
set shown at mistdb.com/mist/genomes/GCF_000006805.1/signal-genes
?kind=output&function=DNA%20binding.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://mib-jouline-db.asc.ohio-state.edu/v1"
GENOME_VERSION = "GCF_000006805.1"
TAXON_ID = 64091
KIND = "output"
FUNCTION = "DNA binding"
PAGE_SIZE = 100
OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "halobacterium_salinarum_nrc1_mist_dna_binding_genes.csv"
)


def api_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def check_genome(session):
    response = session.get(
        f"{API_URL}/genomes",
        params={"search": GENOME_VERSION, "per_page": 10},
        timeout=60,
    )
    response.raise_for_status()
    genome = next(
        (row for row in response.json() if row["version"] == GENOME_VERSION),
        None,
    )
    if genome is None:
        raise ValueError(f"MiST4 has no genome {GENOME_VERSION}")
    if genome["taxonomy_id"] != TAXON_ID:
        raise ValueError(f"{GENOME_VERSION} is taxon {genome['taxonomy_id']}")
    return genome


def dna_binding_domains(session):
    """Domain names MiST labels as output / DNA binding."""
    response = session.get(
        f"{API_URL}/signal_domains",
        params={"per_page": 500},
        timeout=120,
    )
    response.raise_for_status()
    return {
        domain["name"]
        for domain in response.json()
        if domain["kind"] == KIND and domain["function"] == FUNCTION
    }


def download_signal_genes(session):
    rows, page = [], 1
    while True:
        response = session.get(
            f"{API_URL}/genomes/{GENOME_VERSION}/signal-genes",
            params={
                "kind": KIND,
                "function": FUNCTION,
                "fields.Gene": "locus,old_locus,product,version,aseq_id",
                "per_page": PAGE_SIZE,
                "page": page,
            },
            timeout=120,
        )
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        page += 1


def clean_rows(genes, qualifying_domains, genome, retrieved_at):
    cleaned = []
    for gene in genes:
        record = gene.get("Gene") or {}
        locus = record.get("locus")
        if not locus:
            raise ValueError(f"MiST signal gene {gene['id']} has no locus tag")
        domains = sorted(set(gene.get("outputs") or []) & qualifying_domains)
        if not domains:
            raise ValueError(f"{locus} has no DNA-binding output domain")
        cleaned.append({
            "locus_tag": locus,
            "old_locus_tag": record.get("old_locus") or "",
            "refseq_protein": record.get("version") or "",
            "mist_aseq_id": record.get("aseq_id") or "",
            "mist_product": record.get("product") or "",
            "mist_ranks": "|".join(gene.get("ranks") or []),
            "mist_dna_binding_domains": "|".join(domains),
            "mist_input_domains": "|".join(gene.get("inputs") or []),
            "mist_output_domains": "|".join(gene.get("outputs") or []),
            "mist_signal_gene_id": gene["id"],
            "mist_gene_id": gene["gene_id"],
            "genome_version": genome["version"],
            "genome_name": genome["name"],
            "taxonomy_id": genome["taxonomy_id"],
            "selection": f"kind={KIND}; function={FUNCTION}",
            "retrieved_at_utc": retrieved_at,
        })

    loci = [row["locus_tag"] for row in cleaned]
    if len(loci) != len(set(loci)):
        raise ValueError("MiST4 returned duplicate locus tags")
    if not cleaned:
        raise ValueError("MiST4 returned no DNA-binding signal genes")
    return sorted(cleaned, key=lambda row: row["locus_tag"])


def main():
    session = api_session()
    genome = check_genome(session)
    qualifying = dna_binding_domains(session)
    genes = download_signal_genes(session)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cleaned = clean_rows(genes, qualifying, genome, retrieved_at)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    domain_hits = sum(
        len(row["mist_dna_binding_domains"].split("|")) for row in cleaned
    )
    print(f"MiST4 genome: {genome['name']} ({genome['version']})")
    print(f"Output / DNA-binding domain definitions: {len(qualifying)}")
    print(f"DNA-binding signal genes: {len(cleaned)}")
    print(f"Distinct DNA-binding domains per gene, summed: {domain_hits}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
