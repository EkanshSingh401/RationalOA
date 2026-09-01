#!/usr/bin/env python3
"""Print the raw gt_parse shape, then run Experiment 0: check whether the
fake-w2-us-tax-form-dataset records satisfy basic W-2 arithmetic.

  Formula A: Box4 - 0.062 * (Box3 + Box7)
  Formula B: Box5 - Box1 - Sum(Box12 deferral-code amounts)

Run: python3 scripts/check_dataset_consistency.py
"""

import statistics
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w2.datasets import DATASET_NAME, DEFERRAL_BOX12_CODES, from_gt_parse, iter_raw_records

# The dataset has no tax-year field; see the "no tax year" note in the report.
PLACEHOLDER_TAX_YEAR = 2024

RELEVANT_FIELDS = (
    "wages_box1", "ss_wages_box3", "ss_tax_box4", "medicare_wages_box5", "ss_tips_box7",
)


def cents_to_dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def print_raw_shape() -> None:
    print("=" * 70)
    print(f"Raw record shape -- {DATASET_NAME}")
    print("=" * 70)
    _, _, raw = next(iter_raw_records(("train",)))
    print(f"gt_parse key count: {len(raw)}")
    print()
    for key in sorted(raw):
        value = raw[key]
        print(f"  {key:45s} {type(value).__name__:6s} {value!r}")
    print()


def report_check(label: str, diffs_cents: list) -> None:
    n = len(diffs_cents)
    within_1 = sum(1 for d in diffs_cents if abs(d) <= 100)
    dollars = sorted(d / 100 for d in diffs_cents)

    print(f"-- {label} --")
    print(f"  within $1:  {within_1}/{n}  ({within_1 / n:.1%})")
    print(f"  mean:       {cents_to_dollars(round(statistics.mean(diffs_cents)))}")
    print(f"  median:     {cents_to_dollars(round(statistics.median(diffs_cents)))}")
    print(f"  stdev:      {cents_to_dollars(round(statistics.pstdev(diffs_cents)))}")
    print(f"  min:        {cents_to_dollars(min(diffs_cents))}")
    print(f"  max:        {cents_to_dollars(max(diffs_cents))}")
    for p in (1, 5, 25, 50, 75, 95, 99):
        idx = min(n - 1, int(n * p / 100))
        print(f"  p{p:<3d}:       ${dollars[idx]:,.2f}")
    print()


def run_experiment_0() -> None:
    print("=" * 70)
    print("Experiment 0: Box4 / Box5 arithmetic consistency (2,000 records)")
    print("=" * 70)
    print(f"Deferral Box 12 codes used for Formula B: {sorted(DEFERRAL_BOX12_CODES)}")
    print()

    diff_a, diff_b = [], []
    missing = {name: 0 for name in RELEVANT_FIELDS}
    n = 0

    for _, _, raw in iter_raw_records():
        n += 1
        record = from_gt_parse(raw, tax_year=PLACEHOLDER_TAX_YEAR)
        flat = record.flat_fields()

        for name in RELEVANT_FIELDS:
            f = flat.get(name)
            if f is None or f.value is None:
                missing[name] += 1

        box1 = flat["wages_box1"].value
        box3 = flat["ss_wages_box3"].value
        box4 = flat["ss_tax_box4"].value
        box5 = flat["medicare_wages_box5"].value
        box7_field = flat.get("ss_tips_box7")
        box7 = box7_field.value if box7_field and box7_field.value is not None else 0

        expected_a = (Decimal(box3 + box7) * Decimal("0.062")).to_integral_value()
        diff_a.append(box4 - int(expected_a))

        deferral_sum = sum(
            e.amount.value
            for e in record.box12
            if e.code.value in DEFERRAL_BOX12_CODES and e.amount.value is not None
        )
        diff_b.append(box5 - box1 - deferral_sum)

    report_check("Formula A: Box4 - 0.062*(Box3+Box7)", diff_a)
    report_check("Formula B: Box5 - Box1 - Sum(deferral Box12)", diff_b)

    print("Missing-field counts (fields feeding the two formulas above):")
    for name in RELEVANT_FIELDS:
        print(f"  {name:20s} missing in {missing[name]}/{n}")
    print()


if __name__ == "__main__":
    print_raw_shape()
    run_experiment_0()
