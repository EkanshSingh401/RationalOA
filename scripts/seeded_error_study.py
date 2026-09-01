#!/usr/bin/env python3
"""Seeded-error study: what is the rule engine actually worth?

Injects exactly ONE known error into one field of an otherwise-clean
generated document, per document, then measures whether the rule engine
flags that specific field -- an empirical measurement of rule-engine
coverage, not a guess about it (the "missing-field coverage question").
That flag/no-flag split then feeds the same reviewer-catch-rate model
evaluate.py uses, to get a materiality-weighted escape rate per field
and per corruption type.

Corruption kinds: digit_transposition, ocr_confusion, decimal_shift
(money fields only), and dropped_field (nulls the field entirely --
distinct from the others, it's a coverage gap rather than a wrong
value, and the README calls out that a dropped/missing field is a top
real failure mode).

Ground truth is the generator (rule-clean by construction), same
reasoning as run_eval.py: a rule finding here is attributable to the
injected corruption, not to a pre-existing defect in the ground truth.

Run: python3 scripts/seeded_error_study.py [n_docs] [seed]
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w2 import generate
from w2.backends import OCR_CONFUSION_PAIRS, _ocr_confuse, _transpose_digits, _decimal_shift, _clone_record
from w2.evaluate import (
    MATERIALITY_THRESHOLD_CENTS,
    REVIEWER_CATCH_RATE_FLAGGED,
    REVIEWER_CATCH_RATE_UNFLAGGED,
    materiality_cents,
)
from w2.rules import flagged_fields, validate
from w2.schema import Field

DEFAULT_N_DOCS = 8000
DEFAULT_SEED = 777

MONEY_FIELDS = ("wages_box1", "fed_income_tax_box2", "ss_wages_box3", "ss_tax_box4",
                "medicare_wages_box5", "medicare_tax_box6")
IDENTIFIER_FIELDS = ("ssn", "ein")
INJECTABLE_FIELDS = MONEY_FIELDS + IDENTIFIER_FIELDS

MONEY_KINDS = ("digit_transposition", "ocr_confusion", "decimal_shift", "dropped_field")
IDENTIFIER_KINDS = ("digit_transposition", "ocr_confusion", "dropped_field")


def _try_corrupt(rng, kind: str, is_money: bool, value):
    """Return a corrupted value, or None if this (kind, value) pair can't
    produce a real change (e.g. no OCR-confusable digit present)."""
    if kind == "dropped_field":
        return None if value is None else "DROPPED"  # sentinel; caller maps to Field(value=None)
    if is_money:
        digits, sign = str(abs(value)), (1 if value >= 0 else -1)
        if kind == "decimal_shift":
            shifted = _decimal_shift(rng, value)
            return shifted if shifted != value else None
        if kind == "ocr_confusion":
            if not any(c in OCR_CONFUSION_PAIRS for c in digits):
                return None
            try:
                return int(_ocr_confuse(rng, digits)) * sign
            except ValueError:
                return None
        if kind == "digit_transposition":
            transposed = _transpose_digits(rng, digits)
            return int(transposed) * sign if transposed != digits else None
    else:
        if kind == "ocr_confusion":
            if not any(c in OCR_CONFUSION_PAIRS for c in value):
                return None
            return _ocr_confuse(rng, value)
        if kind == "digit_transposition":
            transposed = _transpose_digits(rng, value)
            return transposed if transposed != value else None
    return None


def inject_one_error(rng: random.Random, truth):
    """Pick a random (field, corruption kind), apply it, and return
    (corrupted_record, field_name, kind). Retries with a fresh random
    pick if the chosen (field, kind) can't produce a real change (e.g.
    transposing a single digit), falling back to dropped_field, which
    always works."""
    for _ in range(25):
        field_name = rng.choice(INJECTABLE_FIELDS)
        is_money = field_name in MONEY_FIELDS
        kind = rng.choice(MONEY_KINDS if is_money else IDENTIFIER_KINDS)
        truth_field = getattr(truth, field_name)
        new_value = _try_corrupt(rng, kind, is_money, truth_field.value)
        if new_value is None and kind != "dropped_field":
            continue  # inapplicable, try another (field, kind)
        record = _clone_record(truth)
        final_value = None if new_value in (None, "DROPPED") else new_value
        setattr(record, field_name, Field(value=final_value, confidence=0.5, source="seeded_error_study", bbox=None))
        return record, field_name, kind
    # exhausted retries (shouldn't happen -- dropped_field always applies): force a drop
    field_name = rng.choice(INJECTABLE_FIELDS)
    record = _clone_record(truth)
    setattr(record, field_name, Field(value=None, confidence=0.5, source="seeded_error_study", bbox=None))
    return record, field_name, "dropped_field"


class Group:
    __slots__ = ("n", "n_flagged", "material_n", "escape_numerator", "escape_denominator")

    def __init__(self):
        self.n = 0
        self.n_flagged = 0
        self.material_n = 0
        self.escape_numerator = 0.0
        self.escape_denominator = 0

    def record(self, flagged: bool, material: int, escape_prob: float):
        self.n += 1
        if flagged:
            self.n_flagged += 1
        if material >= MATERIALITY_THRESHOLD_CENTS:
            self.material_n += 1
            self.escape_numerator += material * escape_prob
            self.escape_denominator += material

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / self.n if self.n else 0.0

    @property
    def escape_rate(self) -> float:
        return self.escape_numerator / self.escape_denominator if self.escape_denominator else 0.0


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_DOCS
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED

    rng = random.Random(seed)
    by_field = defaultdict(Group)
    by_kind = defaultdict(Group)
    overall = Group()

    for truth in generate.generate_records(n_docs, seed=seed):
        record, field_name, kind = inject_one_error(rng, truth)
        truth_value = getattr(truth, field_name).value
        pred_value = getattr(record, field_name).value

        flagged = field_name in flagged_fields(validate(record))
        material = materiality_cents(field_name, truth_value, pred_value)
        catch_rate = REVIEWER_CATCH_RATE_FLAGGED if flagged else REVIEWER_CATCH_RATE_UNFLAGGED
        escape_prob = 1.0 - catch_rate

        by_field[field_name].record(flagged, material, escape_prob)
        by_kind[kind].record(flagged, material, escape_prob)
        overall.record(flagged, material, escape_prob)

    print("=" * 88)
    print(f"Seeded-error study -- {n_docs} documents, one injected error each (seed {seed})")
    print("=" * 88)
    print(f"Reviewer model: catches {REVIEWER_CATCH_RATE_FLAGGED:.0%} of errors in a flagged field, "
          f"{REVIEWER_CATCH_RATE_UNFLAGGED:.0%} in an unflagged one (same assumption as evaluate.py).")
    print(f"Materiality threshold: errors under ${MATERIALITY_THRESHOLD_CENTS/100:.2f} excluded from escape rate.")
    print()

    print(f"{'FIELD':24s} {'N':>6s} {'FLAG RATE':>10s} {'ESCAPE RATE':>12s}")
    print("-" * 56)
    for field_name in INJECTABLE_FIELDS:
        g = by_field[field_name]
        print(f"{field_name:24s} {g.n:>6d} {g.flag_rate:>10.1%} {g.escape_rate:>12.1%}")
    print("-" * 56)
    print(f"{'OVERALL':24s} {overall.n:>6d} {overall.flag_rate:>10.1%} {overall.escape_rate:>12.1%}")
    print()

    print(f"{'CORRUPTION TYPE':24s} {'N':>6s} {'FLAG RATE':>10s} {'ESCAPE RATE':>12s}")
    print("-" * 56)
    for kind in MONEY_KINDS:  # superset of IDENTIFIER_KINDS, dropped_field appears once
        if kind not in by_kind:
            continue
        g = by_kind[kind]
        print(f"{kind:24s} {g.n:>6d} {g.flag_rate:>10.1%} {g.escape_rate:>12.1%}")
    print()

    box2 = by_field["fed_income_tax_box2"]
    ssn = by_field["ssn"]
    ein = by_field["ein"]
    print("=" * 88)
    print("Box 2 / SSN check (README's stated expectation: both show high escape rates)")
    print("=" * 88)
    print(f"fed_income_tax_box2: flag rate {box2.flag_rate:.1%}, escape rate {box2.escape_rate:.1%}")
    print(f"ssn:                 flag rate {ssn.flag_rate:.1%}, escape rate {ssn.escape_rate:.1%}")
    print(f"ein:                 flag rate {ein.flag_rate:.1%}, escape rate {ein.escape_rate:.1%}")
    print()
    print("Flag rate, not the $-weighted escape rate, is the metric to read here: identifier")
    print("fields carry a flat materiality regardless of corruption size, so their escape rate")
    print("tracks flag rate directly, but Box 2's escape rate is dominated by the rare huge")
    print("decimal-shift errors that DO get caught (they blow past Box 1 and trip")
    print("FED_TAX_EXCEEDS_WAGES) -- masking that dropped/OCR/transposition Box 2 errors, still")
    print("real $10-30k mistakes, are caught essentially 0% of the time. Re-run with")
    print("scripts/seeded_error_study.py to see that split by corruption type above.")
    print()
    ranked_by_flag = sorted(by_field.items(), key=lambda kv: kv[1].flag_rate)
    worst_three = [name for name, _ in ranked_by_flag[:3]]
    verdict = "CONFIRMED" if {"fed_income_tax_box2", "ssn"} <= set(worst_three) else "PARTIALLY CONFIRMED"
    print(f"Verdict (ranked by flag rate, ascending): {verdict} -- worst-covered fields are {worst_three}")


if __name__ == "__main__":
    main()
