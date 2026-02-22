#!/usr/bin/env python3
"""Query Allen Cell Types API metadata and compare with local cache counts."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE_API = "https://api.brain-map.org/api/v2/data/query.json"


def query_total_rows(query_stages: list[str], timeout: int = 30) -> int:
    q = ",".join(query_stages)
    encoded = urllib.parse.urlencode({"q": q})
    url = f"{BASE_API}?{encoded}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("success"):
        raise RuntimeError(f"API query failed: {payload}")
    return int(payload["total_rows"])


def celltype_count_query(stages: list[str] | None = None) -> int:
    if stages is None:
        stages = []
    stages = ["model::ApiCellTypesSpecimenDetail"] + stages + ["rma::options[num_rows$eq0]"]
    return query_total_rows(stages)


def local_counts(ephys_csv: Path, cells_json: Path) -> tuple[int, int]:
    with ephys_csv.open(newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        ephys_rows = sum(1 for row in reader if any(cell.strip() for cell in row))

    with cells_json.open() as f:
        cells = json.load(f)
    return ephys_rows, len(cells)


def count_with_filter(field: str, value: str) -> int:
    return celltype_count_query([f"rma::criteria[{field}$eq'{value}']"])


def build_report_lines() -> list[str]:
    full_cells = celltype_count_query([])
    all_specimens = query_total_rows([
        "model::Specimen",
        "rma::options[num_rows$eq0]",
    ])
    full_cells_no_ephys_filter = query_total_rows([
        "model::Specimen",
        "rma::criteria[is_cell_specimen$eq'true']",
        "products[name$in'Mouse Cell Types','Human Cell Types']",
        "rma::options[num_rows$eq0]",
    ])
    ephys_cells = query_total_rows([
        "model::Specimen",
        "rma::criteria[is_cell_specimen$eq'true']",
        "products[name$in'Mouse Cell Types','Human Cell Types']",
        "ephys_result[failed$eqfalse]",
        "rma::options[num_rows$eq0]",
    ])

    mouse = count_with_filter("donor__species", "Mus musculus")
    human = count_with_filter("donor__species", "Homo Sapiens")
    spiny = count_with_filter("tag__dendrite_type", "spiny")
    aspiny = count_with_filter("tag__dendrite_type", "aspiny")
    sparse = count_with_filter("tag__dendrite_type", "sparsely spiny")
    recon = celltype_count_query(["rma::criteria[nr__reconstruction_type$ne'null']"])
    no_recon = celltype_count_query(["rma::criteria[nr__reconstruction_type$eqnull]"])

    return [
        f"Live API total (ApiCellTypesSpecimenDetail): {full_cells}",
        f"Live API total of all Specimen objects: {all_specimens}",
        f"Live API total (is_cell_specimen, Cell Types products, ephys_result failed=false): {ephys_cells}",
        f"Live API total same spec without ephys_result failed filter: {full_cells_no_ephys_filter}",
        f"Species: Mus musculus={mouse}, Homo sapiens={human}",
        f"Dendrite labels: spiny={spiny}, aspiny={aspiny}, sparsely spiny={sparse}",
        f"Reconstruction type present: {recon}, missing: {no_recon}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Allen Cell Types API metadata without downloading full datasets."
    )
    parser.add_argument("--ephys-csv", type=Path, default=Path("scripts/cell_types/ephys_features.csv"))
    parser.add_argument("--cells-json", type=Path, default=Path("scripts/cell_types/cells.json"))
    args = parser.parse_args()

    ephys_rows, cells_rows = local_counts(args.ephys_csv, args.cells_json)
    api_lines = build_report_lines()
    full_cells = celltype_count_query([])

    for line in api_lines:
        print(line)
    print(f"Local ephys rows: {ephys_rows}")
    print(f"Local cells.json rows: {cells_rows}")
    print(f"Exact local cache match to live API ApiCellTypesSpecimenDetail total: {ephys_rows == cells_rows == full_cells}")


if __name__ == "__main__":
    main()
