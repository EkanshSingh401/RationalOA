"""Paired shadow evaluation harness.

Every arm scores every document (shadow, not split traffic) -- documents
are cheap to re-score and this gives every metric below the same
denominator across arms, so differences are paired and comparisons don't
need wide confidence intervals to be trustworthy.

Error classification is done by diffing truth against prediction
(classify_field_diff) rather than by asking the backend what it did to
itself. That's deliberate: it's the only approach that works identically
for a simulated backend and a real one, which has no way to "confess"
what it got wrong.

--- Reviewer model: two constants that drive the headline metric ---

REVIEWER_CATCH_RATE_FLAGGED and REVIEWER_CATCH_RATE_UNFLAGGED encode
automation bias: a reviewer facing a rule-flagged field reads it
carefully and catches most errors; a reviewer facing an unflagged field
in an otherwise-clean-looking form catches few. These two numbers are
ASSUMPTIONS, not measurements -- they are the least defensible numbers
in the project. scripts/seeded_error_study.py exists to replace them
with measured values; until that's done, treat every escape number here
as conditional on these constants being roughly right.

--- Escape metrics: three views, different denominators ---

expected_escapes is a raw count: the expected number of material errors
(materiality >= MATERIALITY_THRESHOLD_CENTS) that survive review under
the reviewer model above. It's an expectation, not a stochastic count --
REVIEWER_CATCH_RATE_* are probabilities, summed rather than sampled, so
the number can be fractional (e.g. 187.3 expected escapes).

  - escape_rate_per_document = expected_escapes / n_docs.
    THE HEADLINE. "How many material errors survive review, per
    document scored" -- comparable across arms with different error
    rates or different documents-per-arm, and answers the question the
    README actually cares about (reviewer-minutes and escape rate are
    both per-document quantities).
  - escape_rate_per_error = expected_escapes / n_material_errors.
    Secondary. "Given a material error occurred, what's its average
    chance of surviving review" -- unweighted by dollar size.
  - materiality_weighted_escape_rate = sum(materiality_i * escape_prob_i)
    / sum(materiality_i), i.e. denominator is total dollars in material
    errors, NOT n_docs and NOT n_material_errors. Secondary diagnostic:
    "of the dollars at risk, what fraction escape." Kept because it's a
    real question, but it can disagree sharply with escape_rate_per_error
    when a few huge, easily-caught errors dominate the dollar total and
    mask that most individual errors of that type go uncaught -- this
    happened with Box 2 decimal-shift errors in the seeded-error study
    (see scripts/seeded_error_study.py's Box 2 / SSN section) and is
    exactly why this is not the headline.
"""

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from w2.backends import OCR_CONFUSION_PAIRS
from w2.rules import flagged_fields, validate
from w2.schema import CRITICAL_FIELDS, W2Record

# Reviewer model -- see module docstring. Both are assumptions pending
# scripts/seeded_error_study.py measuring the real numbers.
REVIEWER_CATCH_RATE_FLAGGED = 0.80
REVIEWER_CATCH_RATE_UNFLAGGED = 0.20

# Review-time model -- also an assumption, same caveat.
REVIEWER_BASE_MINUTES = 1.5
REVIEWER_MINUTES_PER_FLAGGED_FIELD = 0.6

MATERIALITY_THRESHOLD_CENTS = 100  # errors under $1 are not escapes

# A dollar-equivalent severity for identifier fields (SSN/EIN/names),
# which have no dollar amount of their own but a wrong one misfiles a
# return -- treated as material regardless of the (nonexistent) $ delta.
IDENTIFIER_MATERIALITY_CENTS = 100_000

MONEY_FIELD_PREFIXES = ("wages_box1", "fed_income_tax_box2", "ss_wages_box3", "ss_tax_box4",
                         "medicare_wages_box5", "medicare_tax_box6", "ss_tips_box7",
                         "allocated_tips_box8", "dependent_care_box10", "nonqualified_plans_box11")


def _is_money_field(name: str) -> bool:
    if name in MONEY_FIELD_PREFIXES:
        return True
    return name.startswith("box12[") or "_box16_wages" in name or "_box17_tax" in name


def materiality_cents(field_name: str, truth_value: Any, pred_value: Any) -> int:
    """Dollar-equivalent size of an error, used to weight escape rate."""
    if truth_value == pred_value:
        return 0
    if _is_money_field(field_name):
        tv = truth_value if isinstance(truth_value, int) else 0
        pv = pred_value if isinstance(pred_value, int) else 0
        return abs(pv - tv)
    return IDENTIFIER_MATERIALITY_CENTS


def classify_field_diff(field_name: str, truth_value: Any, pred_value: Any) -> str:
    """Classify an observed truth/prediction mismatch into a corruption
    taxonomy bucket. Diff-based, not backend-reported -- see module
    docstring."""
    if truth_value == pred_value:
        return "none"
    if pred_value is None and truth_value is not None:
        if field_name.startswith("box12["):
            return "dropped_box12_row"
        if field_name.startswith("state["):
            return "dropped_state_field"  # see row_event_taxonomy for the row-level "dropped_second_state_row" count
        return "dropped_field"
    if truth_value is None and pred_value is not None:
        if field_name.startswith("box12["):
            return "hallucinated_box12_row"
        return "hallucinated_field"
    if isinstance(truth_value, str) and isinstance(pred_value, str):
        if len(truth_value) == len(pred_value) and sorted(truth_value) == sorted(pred_value):
            return "digit_transposition"
        if len(truth_value) == len(pred_value):
            diffs = [(a, b) for a, b in zip(truth_value, pred_value) if a != b]
            if len(diffs) == 1 and OCR_CONFUSION_PAIRS.get(diffs[0][0]) == diffs[0][1]:
                return "ocr_confusion"
        return "identifier_mismatch"
    if isinstance(truth_value, int) and isinstance(pred_value, int):
        if truth_value != 0:
            ratio = pred_value / truth_value
            for shift in (10, 100, 0.1, 0.01):
                if abs(ratio - shift) < 1e-6:
                    return "decimal_shift"
        truth_digits, pred_digits = str(abs(truth_value)), str(abs(pred_value))
        if len(truth_digits) == len(pred_digits) and sorted(truth_digits) == sorted(pred_digits):
            return "digit_transposition"
        return "amount_mismatch"
    return "other"


def _row_group_diff(truth_flat: Dict[str, Any], pred_flat: Dict[str, Any], prefix: str) -> List[str]:
    """Row-level (not field-level) taxonomy entries for box12[..]/state[..]
    groups that vanished or appeared wholesale, so a dropped/hallucinated
    row is counted once, not once per sub-field."""
    def group_keys(flat, marker):
        groups = defaultdict(list)
        for key in flat:
            if key.startswith(marker):
                groups[key.split("]")[0] + "]"].append(key)
        return groups

    events = []
    truth_groups = group_keys(truth_flat, prefix)
    pred_groups = group_keys(pred_flat, prefix)
    for group in set(truth_groups) - set(pred_groups):
        events.append("dropped_box12_row" if prefix == "box12[" else "dropped_second_state_row")
    for group in set(pred_groups) - set(truth_groups):
        events.append("hallucinated_box12_row" if prefix == "box12[" else "hallucinated_state_row")
    return events


@dataclass
class FieldScore:
    field: str
    correct: bool
    confidence: float
    materiality: int
    flagged: bool
    corruption_type: str


@dataclass
class DocScore:
    index: int
    field_scores: List[FieldScore]
    row_events: List[str]
    cost_cents: int
    latency_ms: float
    num_flagged: int


def score_document(index: int, truth: W2Record, pred: W2Record, cost_cents: int, latency_ms: float) -> DocScore:
    truth_flat = {k: v.value for k, v in truth.flat_fields().items()}
    pred_flat = {k: v.value for k, v in pred.flat_fields().items()}
    pred_conf = {k: v.confidence for k, v in pred.flat_fields().items()}

    flagged = flagged_fields(validate(pred))

    row_events = _row_group_diff(truth_flat, pred_flat, "box12[") + _row_group_diff(truth_flat, pred_flat, "state[")

    scores = []
    for key in set(truth_flat) | set(pred_flat):
        tv, pv = truth_flat.get(key), pred_flat.get(key)
        correct = tv == pv
        scores.append(FieldScore(
            field=key,
            correct=correct,
            confidence=pred_conf.get(key, 0.0),
            materiality=materiality_cents(key, tv, pv),
            flagged=key in flagged,
            corruption_type="none" if correct else classify_field_diff(key, tv, pv),
        ))
    return DocScore(index, scores, row_events, cost_cents, latency_ms, len(flagged))


# ---------------------------------------------------------------------------
# risk-coverage
# ---------------------------------------------------------------------------

def risk_coverage_curve(confidences: List[float], corrects: List[bool]):
    """Sort by confidence descending; risk(coverage) = error rate among
    the top `coverage` fraction of items by confidence. Returns a list of
    (coverage, risk) points. Lower risk at a given coverage is better."""
    n = len(confidences)
    if n == 0:
        return [(0.0, 0.0)]
    order = sorted(range(n), key=lambda i: -confidences[i])
    curve = []
    wrong = 0
    for rank, i in enumerate(order, start=1):
        if not corrects[i]:
            wrong += 1
        curve.append((rank / n, wrong / rank))
    return curve


def rc_auc(curve) -> float:
    """Trapezoidal area under the risk-coverage curve. Lower is better --
    it means errors concentrate at low confidence, so accepting the
    highest-confidence slice first stays low-risk for longer."""
    auc = 0.0
    prev_cov, prev_risk = 0.0, curve[0][1] if curve else 0.0
    for cov, risk in curve:
        auc += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return auc


# ---------------------------------------------------------------------------
# per-arm summary
# ---------------------------------------------------------------------------

@dataclass
class ArmReport:
    name: str
    n_docs: int
    field_accuracy: float
    critical_field_accuracy: float
    n_material_errors: int
    expected_escapes: float                # raw count (expected value, not a materiality-weighted probability)
    escape_rate_per_document: float        # HEADLINE: expected_escapes / n_docs
    escape_rate_per_error: float           # secondary: expected_escapes / n_material_errors (unweighted)
    materiality_weighted_escape_rate: float  # secondary: same ratio as before -- dollar-weighted, denominator is
                                              # sum(materiality of material errors), NOT n_docs and NOT a plain
                                              # error count. Kept because "what fraction of dollars at risk
                                              # escape" is a real question; just not the headline anymore.
    materiality_at_risk_cents: int
    reviewer_minutes_per_doc: float
    mean_cost_cents: float
    p95_latency_ms: float
    rc_auc: float
    error_taxonomy: Counter
    row_event_taxonomy: Counter


def _escape_probability(flagged: bool) -> float:
    catch_rate = REVIEWER_CATCH_RATE_FLAGGED if flagged else REVIEWER_CATCH_RATE_UNFLAGGED
    return 1.0 - catch_rate


def summarize_arm(name: str, docs: List[DocScore]) -> ArmReport:
    all_scores = [fs for d in docs for fs in d.field_scores]
    n_fields = len(all_scores)
    n_correct = sum(1 for fs in all_scores if fs.correct)

    critical_scores = [fs for fs in all_scores if fs.field in CRITICAL_FIELDS]
    n_critical_correct = sum(1 for fs in critical_scores if fs.correct)

    material_errors = [fs for fs in all_scores if not fs.correct and fs.materiality >= MATERIALITY_THRESHOLD_CENTS]
    n_material_errors = len(material_errors)

    # expected_escapes is a raw count (expected value under the reviewer model,
    # since REVIEWER_CATCH_RATE_* are probabilities, not a stochastic draw per error).
    escape_probs = [_escape_probability(fs.flagged) for fs in material_errors]
    expected_escapes = sum(escape_probs)

    # Dollar-weighted view: same ratio the module used to call the headline.
    # Denominator is sum(materiality), NOT n_docs and NOT n_material_errors --
    # kept as a secondary diagnostic, see the ArmReport field comment.
    mat_numerator = sum(fs.materiality * prob for fs, prob in zip(material_errors, escape_probs))
    mat_denominator = sum(fs.materiality for fs in material_errors)

    reviewer_minutes = statistics.mean(
        REVIEWER_BASE_MINUTES + REVIEWER_MINUTES_PER_FLAGGED_FIELD * d.num_flagged for d in docs
    ) if docs else 0.0

    mean_cost = statistics.mean(d.cost_cents for d in docs) if docs else 0.0
    latencies = sorted(d.latency_ms for d in docs)
    p95_latency = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    curve = risk_coverage_curve([fs.confidence for fs in all_scores], [fs.correct for fs in all_scores])

    taxonomy = Counter(fs.corruption_type for fs in all_scores if not fs.correct)
    row_taxonomy = Counter(event for d in docs for event in d.row_events)

    return ArmReport(
        name=name,
        n_docs=len(docs),
        field_accuracy=n_correct / n_fields if n_fields else 0.0,
        critical_field_accuracy=n_critical_correct / len(critical_scores) if critical_scores else 0.0,
        n_material_errors=n_material_errors,
        expected_escapes=expected_escapes,
        escape_rate_per_document=(expected_escapes / len(docs)) if docs else 0.0,
        escape_rate_per_error=(expected_escapes / n_material_errors) if n_material_errors else 0.0,
        materiality_weighted_escape_rate=(mat_numerator / mat_denominator) if mat_denominator else 0.0,
        materiality_at_risk_cents=int(mat_denominator),
        reviewer_minutes_per_doc=reviewer_minutes,
        mean_cost_cents=mean_cost,
        p95_latency_ms=p95_latency,
        rc_auc=rc_auc(curve),
        error_taxonomy=taxonomy,
        row_event_taxonomy=row_taxonomy,
    )


def run_shadow_eval(truths: List[W2Record], arms: Dict[str, Any], rng) -> Dict[str, ArmReport]:
    """Score every arm against every document (paired). `arms` values
    must implement extract_from_truth(record, rng) -- real Backend
    instances would instead be called via extract(image) once wired up;
    only the call site here would change."""
    per_arm_docs: Dict[str, List[DocScore]] = {name: [] for name in arms}
    for i, truth in enumerate(truths):
        for name, backend in arms.items():
            latency = backend.sample_latency_ms(rng)
            pred = backend.extract_from_truth(truth, rng)
            per_arm_docs[name].append(score_document(i, truth, pred, backend.cost_cents_per_doc, latency))
    return {name: summarize_arm(name, docs) for name, docs in per_arm_docs.items()}


def find_calibration_inversions(reports: Dict[str, ArmReport]) -> List[str]:
    """An inversion: arm A has higher field accuracy than arm B, but a
    WORSE (higher) RC-AUC. Under mandatory sign-off, confidence is what
    rations reviewer attention -- selecting on field accuracy alone picks
    the wrong arm in this case."""
    messages = []
    names = list(reports)
    for a in names:
        for b in names:
            if a == b:
                continue
            ra, rb = reports[a], reports[b]
            if ra.field_accuracy > rb.field_accuracy and ra.rc_auc > rb.rc_auc:
                messages.append(
                    f"INVERSION: '{a}' has higher field accuracy than '{b}' "
                    f"({ra.field_accuracy:.1%} vs {rb.field_accuracy:.1%}) but WORSE "
                    f"calibration (RC-AUC {ra.rc_auc:.4f} vs {rb.rc_auc:.4f}, lower is better). "
                    f"Under mandatory sign-off, confidence is what rations reviewer attention -- "
                    f"selecting '{a}' on field accuracy alone picks the wrong arm."
                )
    return messages
