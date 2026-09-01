import copy
import random

import pytest

from w2 import generate, rules
from w2.backends import ARMS
from w2.constants import ty
from w2.evaluate import (
    MATERIALITY_THRESHOLD_CENTS,
    REVIEWER_CATCH_RATE_FLAGGED,
    REVIEWER_CATCH_RATE_UNFLAGGED,
    rc_auc,
    risk_coverage_curve,
    run_shadow_eval,
    score_document,
    summarize_arm,
)
from w2.schema import Field, W2Record


def f(value, confidence=1.0, source="test"):
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def make_record(tax_year=2024, **overrides):
    consts = ty(tax_year)
    box1 = box5 = box3 = 100_000_00
    box7 = 0
    record = W2Record(
        ssn=f("123-45-6789"),
        ein=f("12-3456789"),
        tax_year=f(tax_year),
        employer_name=f("Acme Corp"),
        employee_name=f("Jane Doe"),
        wages_box1=f(box1),
        fed_income_tax_box2=f(15_000_00),
        ss_wages_box3=f(box3),
        ss_tax_box4=f(rules.expected_ss_tax_cents(box3, box7, consts)),
        medicare_wages_box5=f(box5),
        medicare_tax_box6=f(rules.expected_medicare_tax_cents(box5, consts)),
        ss_tips_box7=f(box7),
        allocated_tips_box8=f(0),
        dependent_care_box10=f(0),
        nonqualified_plans_box11=f(0),
        box13_retirement_plan=f(False),
        box12=[],
        state_rows=[],
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


# --- pinning the metric definitions ---------------------------------------

def test_two_known_errors_one_flagged_one_not():
    """Hand-built truth/prediction pair with exactly 2 errors: a wrong
    Box 4 (SS_TAX_MISMATCH will flag ss_tax_box4) and a wrong Box 2 (no
    rule covers Box 2 -- README's known gap -- so it stays unflagged).
    Pins n_material_errors and expected_escapes to the two reviewer
    constants directly, so a refactor can't silently redefine either."""
    truth = make_record()

    pred = copy.deepcopy(truth)
    pred.ss_tax_box4 = f(1)  # wildly wrong -> SS_TAX_MISMATCH fires
    pred.fed_income_tax_box2 = f(truth.fed_income_tax_box2.value + 500_00)  # still << box1, no rule covers Box 2

    findings = rules.validate(pred)
    flagged = rules.flagged_fields(findings)
    assert "ss_tax_box4" in flagged
    assert "fed_income_tax_box2" not in flagged

    doc = score_document(0, truth, pred, cost_cents=0, latency_ms=0.0)
    report = summarize_arm("test_arm", [doc])

    assert report.n_material_errors == 2

    expected_escapes = (1 - REVIEWER_CATCH_RATE_FLAGGED) + (1 - REVIEWER_CATCH_RATE_UNFLAGGED)
    assert report.expected_escapes == pytest.approx(expected_escapes)
    assert report.escape_rate_per_document == pytest.approx(expected_escapes / 1)
    assert report.escape_rate_per_error == pytest.approx(expected_escapes / 2)


def test_error_under_materiality_floor_excluded():
    truth = make_record()

    pred = copy.deepcopy(truth)
    tiny_diff = MATERIALITY_THRESHOLD_CENTS - 50  # under $1
    assert tiny_diff < MATERIALITY_THRESHOLD_CENTS
    pred.fed_income_tax_box2 = f(truth.fed_income_tax_box2.value + tiny_diff)

    doc = score_document(0, truth, pred, cost_cents=0, latency_ms=0.0)
    report = summarize_arm("test_arm", [doc])

    # the field really is wrong, it's just immaterial
    assert any(fs.field == "fed_income_tax_box2" and not fs.correct for fs in doc.field_scores)
    assert report.n_material_errors == 0
    assert report.expected_escapes == 0.0


def test_perfect_prediction_yields_zero_escapes():
    truth = make_record()
    pred = copy.deepcopy(truth)

    doc = score_document(0, truth, pred, cost_cents=100, latency_ms=500.0)
    report = summarize_arm("test_arm", [doc])

    assert report.field_accuracy == 1.0
    assert report.n_material_errors == 0
    assert report.expected_escapes == 0.0
    assert report.escape_rate_per_document == 0.0


# --- risk-coverage ----------------------------------------------------

def test_perfectly_separated_confidence_beats_uncorrelated():
    n = 100
    correct = [i % 2 == 0 for i in range(n)]  # alternating, 50/50, uncorrelated with index parity trend

    # confidence perfectly separates: every correct item outranks every wrong one
    separated_conf = [0.9 if c else 0.1 for c in correct]

    # confidence uncorrelated with correctness: monotonic ramp against an
    # alternating correctness pattern -- the top-k by confidence is close
    # to 50/50 right/wrong at every coverage level
    uncorrelated_conf = [(i + 1) / n for i in range(n)]

    separated_auc = rc_auc(risk_coverage_curve(separated_conf, correct))
    uncorrelated_auc = rc_auc(risk_coverage_curve(uncorrelated_conf, correct))

    assert separated_auc < uncorrelated_auc


# --- paired evaluation --------------------------------------------------

def test_shadow_eval_is_paired_across_arms():
    truths = list(generate.generate_records(50, seed=3))
    rng = random.Random(9)
    arms = {"hosted_prebuilt": ARMS["hosted_prebuilt"], "template_ocr": ARMS["template_ocr"]}

    reports = run_shadow_eval(truths, arms, rng)

    assert reports["hosted_prebuilt"].n_docs == reports["template_ocr"].n_docs == len(truths)

    # same property at the per-document level, not just the aggregated report
    rng2 = random.Random(9)
    docs_a = [
        score_document(i, t, arms["hosted_prebuilt"].extract_from_truth(t, rng2), 0, 0.0)
        for i, t in enumerate(truths)
    ]
    docs_b = [
        score_document(i, t, arms["template_ocr"].extract_from_truth(t, rng2), 0, 0.0)
        for i, t in enumerate(truths)
    ]
    assert len(docs_a) == len(docs_b) == len(truths)
