#!/usr/bin/env python3
"""Demonstrate the downstream-output stage: export a batch of SIGNED_OFF
W-2s to CSV and JSON, and demonstrate that the sign-off gate is a
mechanical property of the exporter, not a policy someone has to remember.

Uses w2.generate for example data (rule-clean by construction) rather
than a real document store, which this project doesn't have yet -- see
w2/output.py's docstring for the CSV/JSON format details and exactly
why the sign-off and critical-findings gates behave the way they do.

Run: python3 scripts/export.py [n_docs] [seed] [--csv PATH] [--json PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w2 import generate
from w2.chase import DocState
from w2.output import ExportRecord, NotSignedOff, UnresolvedCriticalFindings, export_csv, export_json
from w2.schema import Field


def _signed_off_batch(n_docs: int, seed: int):
    return [
        ExportRecord(doc_id=f"DOC-{i:05d}", state=DocState.SIGNED_OFF, record=record)
        for i, record in enumerate(generate.generate_records(n_docs, seed=seed))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_docs", nargs="?", type=int, default=25)
    parser.add_argument("seed", nargs="?", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=None, help="write CSV export to this path")
    parser.add_argument("--json", type=Path, default=None, help="write JSON export to this path")
    args = parser.parse_args()

    docs = _signed_off_batch(args.n_docs, args.seed)

    csv_text = export_csv(docs)
    json_text = export_json(docs)

    print(f"Exported {len(docs)} SIGNED_OFF documents.")
    print(f"CSV: {len(csv_text.splitlines())} lines (1 header + {len(docs)} rows)")
    print(f"JSON: {len(json_text)} bytes")

    if args.csv:
        args.csv.write_text(csv_text)
        print(f"Wrote CSV to {args.csv}")
    if args.json:
        args.json.write_text(json_text)
        print(f"Wrote JSON to {args.json}")
    if not args.csv and not args.json:
        print("\n--- CSV (first 3 lines) ---")
        print("\n".join(csv_text.splitlines()[:3]))

    print("\n" + "=" * 70)
    print("The sign-off gate is mechanical: a non-SIGNED_OFF record is refused")
    print("=" * 70)
    not_ready = ExportRecord(doc_id="DOC-BAD-STATE", state=DocState.READY_FOR_REVIEW, record=docs[0].record)
    try:
        export_csv([not_ready])
    except NotSignedOff as exc:
        print(f"export_csv refused as expected: {exc}")

    print("\n" + "=" * 70)
    print("A record signed off in error (unresolved CRITICAL finding) -- CSV refuses, JSON flags it")
    print("=" * 70)
    bad_record = docs[0].record
    # simulate a human approving a record with an obviously wrong SS tax withheld
    bad_record.ss_tax_box4 = Field(value=1, confidence=0.9, source="human-override", bbox=None)
    signed_off_in_error = ExportRecord(doc_id="DOC-SIGNED-IN-ERROR", state=DocState.SIGNED_OFF, record=bad_record)
    try:
        export_csv([signed_off_in_error])
    except UnresolvedCriticalFindings as exc:
        print(f"export_csv refused as expected: {exc}")

    flagged_json = export_json([signed_off_in_error])
    flagged = '"has_unresolved_critical_findings": true' in flagged_json
    print(f'export_json still exported it: has_unresolved_critical_findings present = {flagged}')


if __name__ == "__main__":
    main()
