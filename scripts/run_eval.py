#!/usr/bin/env python3
"""Arm comparison table: paired shadow evaluation of the three simulated
backends (w2.backends.ARMS) against a corpus of payroll-consistent
generated ground truth (w2.generate).

Ground truth comes from the generator, not the HF public corpus -- the
generator is rule-clean by construction, so every rule finding observed
here is attributable to a simulated extraction error, not a pre-existing
defect in the ground truth itself (the public corpus's own known FICA
defects would make it useless for this: the rule engine would fire on
it regardless of extraction quality). See w2/generate.py and FINDINGS.md
F1.

Run: python3 scripts/run_eval.py [n_docs] [seed]
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w2 import generate
from w2.backends import ARMS
from w2.evaluate import (
    MATERIALITY_THRESHOLD_CENTS,
    REVIEWER_CATCH_RATE_FLAGGED,
    REVIEWER_CATCH_RATE_UNFLAGGED,
    find_calibration_inversions,
    run_shadow_eval,
)

DEFAULT_N_DOCS = 2000
DEFAULT_SEED = 2024


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_DOCS
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED

    truths = list(generate.generate_records(n_docs, seed=seed))
    rng = random.Random(seed + 1)  # separate stream from the ground-truth generator
    reports = run_shadow_eval(truths, ARMS, rng)

    print("=" * 100)
    print(f"Arm comparison -- {n_docs} paired documents (generator seed {seed})")
    print("=" * 100)
    print(f"Reviewer model (assumption, see w2/evaluate.py docstring): catches "
          f"{REVIEWER_CATCH_RATE_FLAGGED:.0%} of errors in a flagged field, "
          f"{REVIEWER_CATCH_RATE_UNFLAGGED:.0%} in an unflagged one.")
    print(f"Materiality threshold: errors under ${MATERIALITY_THRESHOLD_CENTS/100:.2f} are not escapes.")
    print()

    header = (f"{'ARM':16s} {'FIELD ACC':>10s} {'CRIT ACC':>9s} {'ESC/DOC':>8s} "
              f"{'REV MIN':>8s} {'COST':>8s} {'P95 LAT':>9s} {'RC-AUC':>9s}")
    print(header)
    print("-" * len(header))
    for name, r in reports.items():
        rev_min = f"{r.reviewer_minutes_per_doc:.2f}m"
        cost = "$" + format(r.mean_cost_cents / 100, ".2f")
        latency = f"{r.p95_latency_ms:.0f}ms"
        print(f"{name:16s} {r.field_accuracy:>10.1%} {r.critical_field_accuracy:>9.1%} "
              f"{r.escape_rate_per_document:>8.3f} {rev_min:>8s} "
              f"{cost:>8s} {latency:>9s} {r.rc_auc:>9.4f}")
    print()
    print("ESC/DOC (headline): expected_escapes / n_docs -- material errors surviving review, per document.")
    print("RC-AUC: lower is better (area under the risk-coverage curve -- risk stays low longer")
    print("as low-confidence fields are set aside first, when confidence actually separates errors).")
    print()

    print("=" * 100)
    print("Escape rate -- three views, see w2/evaluate.py docstring for what each denominator means")
    print("=" * 100)
    sec_header = (f"{'ARM':16s} {'N ERRORS':>9s} {'ESCAPES':>9s} {'ESC/DOC':>9s} "
                  f"{'ESC/ERROR':>10s} {'$-WEIGHTED':>11s}")
    print(sec_header)
    print("-" * len(sec_header))
    for name, r in reports.items():
        print(f"{name:16s} {r.n_material_errors:>9d} {r.expected_escapes:>9.1f} "
              f"{r.escape_rate_per_document:>9.3f} {r.escape_rate_per_error:>10.1%} "
              f"{r.materiality_weighted_escape_rate:>11.1%}")
    print()

    inversions = find_calibration_inversions(reports)
    if inversions:
        print("=" * 100)
        print("Calibration inversions")
        print("=" * 100)
        for msg in inversions:
            print(msg)
        print()

    print("=" * 100)
    print("Error taxonomy per arm (top corruption types among incorrect fields)")
    print("=" * 100)
    for name, r in reports.items():
        print(f"\n-- {name} --")
        total_errors = sum(r.error_taxonomy.values())
        for corruption_type, count in r.error_taxonomy.most_common():
            print(f"  {corruption_type:24s} {count:>5d}  ({count/total_errors:.1%} of field errors)" if total_errors else "")
        if r.row_event_taxonomy:
            print(f"  row-level events: {dict(r.row_event_taxonomy)}")


if __name__ == "__main__":
    main()
