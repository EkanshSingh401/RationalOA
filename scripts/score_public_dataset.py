#!/usr/bin/env python3
"""Run the rule engine over all 2,000 singhsays/fake-w2-us-tax-form-dataset
records and report the fire rate per rule, with example records.

KNOWN-DEFECTIVE INPUT, READ BEFORE INTERPRETING THE OUTPUT:
This dataset's amounts are not real payroll data and are known-defective
in specific, characterized ways (see check_dataset_consistency.py):

  - Box 4 is deterministically ~7.65% of Box 3 (the combined employee
    SS+Medicare rate), not the correct 6.2% SS rate -- SS_TAX_MISMATCH
    fires on essentially every record by construction.
  - Box 6 is deterministically ~2.9% of Box 5 (double the correct 1.45%
    Medicare rate) -- MEDICARE_TAX_MISMATCH fires on essentially every
    record by construction.
  - Box 3 is sampled independently of the SS wage base with no cap
    applied. Box 3 alone exceeds the 2024 base in ~38% of records, but
    SS_WAGE_BASE_EXCEEDED checks Box 3 + Box 7 per the rule spec, and
    Box 7 is set identically equal to Box 3 in every record (a separate
    generator artifact -- see check_dataset_consistency.py), so the
    checked total is effectively 2x Box 3. That roughly doubles the
    fire rate to ~78%, not the ~35% you'd get from Box 3 alone -- a
    rule-input artifact, not a rule bug.

A near-100% fire rate on SS_TAX_MISMATCH / MEDICARE_TAX_MISMATCH here is
the expected, correct output of the rule engine -- it is not evidence of
a bug in rules.py. Do not loosen tolerances to make these numbers look
better; see the acceptance test in tests/test_rules.py for what "the
rules are wrong" actually looks like (findings on generator-clean data).

Run: python3 scripts/score_public_dataset.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w2.datasets import DATASET_NAME, from_gt_parse, iter_raw_records
from w2.rules import RULES, validate

PLACEHOLDER_TAX_YEAR = 2024  # dataset carries no year field
EXAMPLES_PER_RULE = 3


def main() -> None:
    hits = defaultdict(list)  # rule_id -> list of (split, index, Finding)
    n = 0

    for split, i, raw in iter_raw_records():
        n += 1
        record = from_gt_parse(raw, tax_year=PLACEHOLDER_TAX_YEAR)
        for finding in validate(record):
            hits[finding.rule_id].append((split, i, finding))

    print("=" * 78)
    print(f"Rule fire rates -- {DATASET_NAME} ({n} records)")
    print("=" * 78)
    print(f"{'RULE_ID':32s} {'SEVERITY':10s} {'FIRE RATE':>12s}")
    print("-" * 78)
    for rule in RULES:
        count = len(hits.get(rule.id, []))
        print(f"{rule.id:32s} {rule.severity.value:10s} {count:>6d}/{n} ({count / n:.1%})")
    print()

    print("=" * 78)
    print("Examples per rule")
    print("=" * 78)
    for rule in RULES:
        records = hits.get(rule.id, [])
        if not records:
            continue
        print(f"\n-- {rule.id} ({len(records)}/{n}) --")
        for split, i, finding in records[:EXAMPLES_PER_RULE]:
            print(f"  [{split}#{i}] {finding.message}")


if __name__ == "__main__":
    main()
