"""Download and validate the Network Portal NRC-1 transcription-factor table."""

import argparse
import csv
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SOURCE_URL = "https://networks.systemsbiology.net/hal/genes/?filter=tf"
EXPECTED_HEADERS = [
    "Name",
    "Common name",
    "Type",
    "Gene ID",
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "Description",
    "TF",
]
OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "halobacterium_salinarum_nrc1_network_portal_tfs.csv"
)


class GeneTableParser(HTMLParser):
    """Extract cells from the table whose HTML id is ``genes-table``."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_cell = False
        self.cell_parts = []
        self.current_row = []
        self.headers = []
        self.rows = []
        self.section = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "genes-table":
            self.in_table = True
        elif self.in_table and tag in {"thead", "tbody"}:
            self.section = tag
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag):
        if self.in_table and tag in {"th", "td"} and self.in_cell:
            value = " ".join("".join(self.cell_parts).split())
            self.current_row.append(value)
            self.in_cell = False
            self.cell_parts = []
        elif self.in_table and tag == "tr" and self.current_row:
            if self.section == "thead":
                self.headers = self.current_row
            elif self.section == "tbody":
                self.rows.append(self.current_row)
            self.current_row = []
        elif self.in_table and tag == "table":
            self.in_table = False
            self.section = None


def download_html():
    session = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    response = session.get(SOURCE_URL, timeout=180)
    response.raise_for_status()
    return response.text


def parse_table(html):
    parser = GeneTableParser()
    parser.feed(html)
    if parser.headers != EXPECTED_HEADERS:
        raise ValueError(
            f"Unexpected Network Portal columns: {parser.headers!r}"
        )

    count_match = re.search(r"transcription factors:\s*(\d+)", html)
    if count_match is None:
        raise ValueError("Network Portal did not report a TF count")
    reported_count = int(count_match.group(1))

    if len(parser.rows) != reported_count:
        raise ValueError(
            f"Expected {reported_count} TF rows, parsed {len(parser.rows)}"
        )
    if any(len(row) != len(EXPECTED_HEADERS) for row in parser.rows):
        raise ValueError("At least one Network Portal row has missing columns")

    records = [dict(zip(EXPECTED_HEADERS, row, strict=True)) for row in parser.rows]
    names = [record["Name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Network Portal returned duplicate TF locus tags")
    if any(record["TF"].lower() != "true" for record in records):
        raise ValueError("Filtered Network Portal table contains a non-TF row")
    return records, reported_count


def clean_records(records, retrieved_at):
    return [
        {
            "locus_tag": record["Name"],
            "common_name": record["Common name"],
            "feature_type": record["Type"],
            "ncbi_gene_id": "" if record["Gene ID"] == "-" else record["Gene ID"],
            "replicon": record["Chromosome"],
            "start": record["Start"],
            "end": record["End"],
            "strand": record["Strand"],
            "description": record["Description"],
            "is_tf": record["TF"].lower(),
            "source_url": SOURCE_URL,
            "retrieved_at_utc": retrieved_at,
        }
        for record in records
    ]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-html",
        type=Path,
        help="Parse a saved page instead of downloading it",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = arguments()
    if args.input_html:
        html = args.input_html.read_text(encoding="utf-8")
    else:
        html = download_html()

    records, reported_count = parse_table(html)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cleaned = clean_records(records, retrieved_at)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Network Portal reported TFs: {reported_count}")
    print(f"Unique TF rows saved: {len(cleaned)}")
    print(f"Saved dataset: {args.output}")


if __name__ == "__main__":
    main()
