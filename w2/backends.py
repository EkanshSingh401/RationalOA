"""Extraction backends.

`Backend` is the real protocol: `extract(image) -> W2Record`. No real
extractor exists yet (Build order step 6) -- every class in this module
is a SIMULATION that stands in for one, driven by an explicit,
documented error model rather than a real model or API. They implement
`extract_from_truth(record, rng) -> W2Record`: take a known-correct
ground-truth record (from w2.generate or w2.datasets) and a seeded RNG,
and return a corrupted copy playing the role of what a real backend
would have extracted from the corresponding image.

`extract_from_truth` is NOT part of the `Backend` protocol -- a real
backend has no ground truth to peek at, so it can never implement this
method. It exists only on SimulatedBackend and is a evaluation-harness
convenience. When a real backend (Template OCR, fine-tuned Donut, hosted
prebuilt, a VLM) is wired up, it implements `extract(image)` only, and
`evaluate.py` / `run_eval.py` swap it in with no other changes -- they
call `backend.extract(image)` for real backends and
`backend.extract_from_truth(record, rng)` for simulated ones, but
everything downstream (scoring, risk-coverage, cost/latency) is
identical either way because it only ever looks at the returned
W2Record.

Simulated error model, applied per Field:
  - digit_transposition   -- two digits in the same value swap places
  - ocr_confusion         -- one character flips within a confusable
                              pair: 5/S, 0/O, 1/7, 8/B
  - decimal_shift         -- a money amount's decimal point moves one or
                              two places (misread magnitude)
  - dropped_second_state_row  -- the second Boxes 15-20 row disappears
  - dropped_box12_row         -- a Box 12 entry disappears
  - hallucinated_box12_row    -- an extra, invented Box 12 entry appears

Three arms are configured, chosen to differ in CALIBRATION as well as
raw accuracy -- see ARMS at the bottom:

  - "hosted_prebuilt": moderate accuracy, well-calibrated confidence
    (its error confidence is clearly lower than its correct confidence).
  - "vlm_structured": the highest raw accuracy of the three, but
    confidence barely separates its own errors from its correct answers
    -- the realistic "badly overconfident VLM" signature the README
    calls out.
  - "template_ocr": cheapest and fastest, but the least accurate.

Confidence is sampled from a clipped Gaussian per Field, using different
(mean, spread) for correct vs. corrupted fields -- the gap between those
two distributions IS the calibration behavior under test.
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Protocol, runtime_checkable

from w2.schema import Box12Entry, Field, StateRow, W2Record

OCR_CONFUSION_PAIRS = {"5": "S", "S": "5", "0": "O", "O": "0", "1": "7", "7": "1", "8": "B", "B": "8"}

CORRUPTION_TYPES = (
    "digit_transposition",
    "ocr_confusion",
    "decimal_shift",
    "dropped_second_state_row",
    "dropped_box12_row",
    "hallucinated_box12_row",
)


@runtime_checkable
class Backend(Protocol):
    """What every real extractor implements. Nothing in this module
    implements it yet -- see the module docstring."""

    name: str

    def extract(self, image: Any) -> W2Record: ...


# ---------------------------------------------------------------------------
# field-level corruption primitives
# ---------------------------------------------------------------------------

def _transpose_digits(rng: random.Random, s: str) -> str:
    idx = [i for i, c in enumerate(s) if c.isdigit()]
    if len(idx) < 2:
        return s
    i, j = rng.sample(idx, 2)
    chars = list(s)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _ocr_confuse(rng: random.Random, s: str) -> str:
    idx = [i for i, c in enumerate(s) if c in OCR_CONFUSION_PAIRS]
    if not idx:
        return s
    i = rng.choice(idx)
    chars = list(s)
    chars[i] = OCR_CONFUSION_PAIRS[chars[i]]
    return "".join(chars)


def _decimal_shift(rng: random.Random, cents: int) -> int:
    shift = rng.choice([-2, -1, 1, 2])
    return int(cents * (10 ** shift)) if shift > 0 else int(cents / (10 ** -shift))


def _corrupt_identifier(rng: random.Random, value: str) -> str:
    """Apply digit transposition or an OCR confusion to an SSN/EIN-shaped
    string. What actually happened is recovered later by diffing truth
    vs. prediction (evaluate.py), not tracked here -- that keeps error
    classification backend-agnostic, so it works on a real backend's
    output too, not just a simulated one's."""
    if rng.random() < 0.5 and any(c in OCR_CONFUSION_PAIRS for c in value):
        return _ocr_confuse(rng, value)
    return _transpose_digits(rng, value)


def _corrupt_money(rng: random.Random, cents: int) -> int:
    """Apply digit transposition, an OCR confusion, or a decimal shift to
    a money amount (cents)."""
    roll = rng.random()
    if roll < 0.35:
        return _decimal_shift(rng, cents)
    digits = str(abs(cents))
    sign = 1 if cents >= 0 else -1
    if roll < 0.70 and any(c in OCR_CONFUSION_PAIRS for c in digits):
        corrupted = _ocr_confuse(rng, digits)
    else:
        corrupted = _transpose_digits(rng, digits)
    try:
        return int(corrupted) * sign
    except ValueError:
        return cents  # confusion produced a non-digit string (e.g. "0" -> "O"); leave uncorrupted


# ---------------------------------------------------------------------------
# error model parameters
# ---------------------------------------------------------------------------

SCALAR_FIELD_ATTRS = (
    "wages_box1", "fed_income_tax_box2", "ss_wages_box3", "ss_tax_box4",
    "medicare_wages_box5", "medicare_tax_box6", "ss_tips_box7",
    "allocated_tips_box8", "dependent_care_box10", "nonqualified_plans_box11",
)
IDENTIFIER_FIELD_ATTRS = ("ssn", "ein")


@dataclass(frozen=True)
class ErrorModel:
    """Parameters for one simulated backend's error/calibration behavior.
    See the module docstring for what each corruption type does."""

    field_error_rate: float          # probability any given scalar/identifier field is corrupted
    drop_second_state_row_rate: float
    drop_box12_row_rate: float
    hallucinate_box12_row_rate: float
    correct_confidence_mean: float
    correct_confidence_spread: float
    error_confidence_mean: float     # how close this is to correct_confidence_mean IS the calibration
    error_confidence_spread: float
    cost_cents_per_doc: int
    latency_ms_mean: float
    latency_ms_spread: float


def _clipped_confidence(rng: random.Random, mean: float, spread: float) -> float:
    return max(0.01, min(0.999, rng.gauss(mean, spread)))


HOSTED_PREBUILT = ErrorModel(
    field_error_rate=0.03,
    drop_second_state_row_rate=0.02,
    drop_box12_row_rate=0.02,
    hallucinate_box12_row_rate=0.005,
    correct_confidence_mean=0.95, correct_confidence_spread=0.04,
    error_confidence_mean=0.55, error_confidence_spread=0.15,
    cost_cents_per_doc=150,
    latency_ms_mean=800, latency_ms_spread=250,
)

VLM_STRUCTURED = ErrorModel(
    field_error_rate=0.015,
    drop_second_state_row_rate=0.01,
    drop_box12_row_rate=0.01,
    hallucinate_box12_row_rate=0.02,
    correct_confidence_mean=0.93, correct_confidence_spread=0.05,
    error_confidence_mean=0.88, error_confidence_spread=0.06,   # barely separated from correct -> overconfident
    cost_cents_per_doc=400,
    latency_ms_mean=1800, latency_ms_spread=500,
)

TEMPLATE_OCR = ErrorModel(
    field_error_rate=0.08,
    drop_second_state_row_rate=0.05,
    drop_box12_row_rate=0.06,
    hallucinate_box12_row_rate=0.01,
    correct_confidence_mean=0.90, correct_confidence_spread=0.07,
    error_confidence_mean=0.45, error_confidence_spread=0.18,
    cost_cents_per_doc=5,
    latency_ms_mean=200, latency_ms_spread=60,
)


# ---------------------------------------------------------------------------
# simulated backend
# ---------------------------------------------------------------------------

class SimulatedBackend:
    """A stand-in extractor driven by an ErrorModel. See the module
    docstring: this does NOT implement the real `Backend.extract(image)`
    protocol, because it has no image -- it corrupts ground truth
    instead, for evaluation purposes only."""

    def __init__(self, name: str, model: ErrorModel):
        self.name = name
        self.model = model

    @property
    def cost_cents_per_doc(self) -> int:
        return self.model.cost_cents_per_doc

    def sample_latency_ms(self, rng: random.Random) -> float:
        return max(1.0, rng.gauss(self.model.latency_ms_mean, self.model.latency_ms_spread))

    def _confidence_for(self, rng: random.Random, is_correct: bool) -> float:
        if is_correct:
            return _clipped_confidence(rng, self.model.correct_confidence_mean, self.model.correct_confidence_spread)
        return _clipped_confidence(rng, self.model.error_confidence_mean, self.model.error_confidence_spread)

    def extract_from_truth(self, record: W2Record, rng: random.Random) -> W2Record:
        result = _clone_record(record)

        for attr in SCALAR_FIELD_ATTRS:
            truth_field = getattr(result, attr)
            if truth_field is None or truth_field.value is None:
                continue
            corrupt = rng.random() < self.model.field_error_rate
            if corrupt:
                value = _corrupt_money(rng, truth_field.value)
            else:
                value = truth_field.value
            confidence = self._confidence_for(rng, not corrupt)
            setattr(result, attr, Field(value=value, confidence=confidence, source=self.name, bbox=None))

        for attr in IDENTIFIER_FIELD_ATTRS:
            truth_field = getattr(result, attr)
            corrupt = rng.random() < self.model.field_error_rate
            if corrupt:
                value = _corrupt_identifier(rng, truth_field.value)
            else:
                value = truth_field.value
            confidence = self._confidence_for(rng, not corrupt)
            setattr(result, attr, Field(value=value, confidence=confidence, source=self.name, bbox=None))

        result.box12 = self._corrupt_box12(rng, result.box12)
        result.state_rows = self._corrupt_state_rows(rng, result.state_rows)

        return result

    def _corrupt_box12(self, rng: random.Random, entries: List[Box12Entry]) -> List[Box12Entry]:
        kept = []
        for entry in entries:
            if rng.random() < self.model.drop_box12_row_rate:
                continue  # dropped_box12_row
            corrupt = rng.random() < self.model.field_error_rate
            if corrupt:
                value = _corrupt_money(rng, entry.amount.value)
            else:
                value = entry.amount.value
            confidence = self._confidence_for(rng, not corrupt)
            kept.append(Box12Entry(
                code=Field(value=entry.code.value, confidence=self._confidence_for(rng, True), source=self.name, bbox=None),
                amount=Field(value=value, confidence=confidence, source=self.name, bbox=None),
            ))
        if rng.random() < self.model.hallucinate_box12_row_rate:
            fake_code = rng.choice(["D", "E", "DD", "W"])
            kept.append(Box12Entry(
                code=Field(value=fake_code, confidence=self._confidence_for(rng, False), source=self.name, bbox=None),
                amount=Field(value=rng.randint(10_00, 500_00), confidence=self._confidence_for(rng, False), source=self.name, bbox=None),
            ))
        return kept

    def _corrupt_state_rows(self, rng: random.Random, rows: List[StateRow]) -> List[StateRow]:
        if len(rows) >= 2 and rng.random() < self.model.drop_second_state_row_rate:
            rows = rows[:1]  # dropped_second_state_row

        result = []
        for row in rows:
            corrupt_wages = rng.random() < self.model.field_error_rate
            wages_value = _corrupt_money(rng, row.state_wages.value) if corrupt_wages else row.state_wages.value
            corrupt_tax = rng.random() < self.model.field_error_rate
            tax_value = _corrupt_money(rng, row.state_income_tax.value) if corrupt_tax else row.state_income_tax.value
            result.append(StateRow(
                state=Field(value=row.state.value, confidence=self._confidence_for(rng, True), source=self.name, bbox=None),
                employer_state_id=Field(value=row.employer_state_id.value, confidence=self._confidence_for(rng, True), source=self.name, bbox=None),
                state_wages=Field(value=wages_value, confidence=self._confidence_for(rng, not corrupt_wages), source=self.name, bbox=None),
                state_income_tax=Field(value=tax_value, confidence=self._confidence_for(rng, not corrupt_tax), source=self.name, bbox=None),
            ))
        return result


def _clone_record(record: W2Record) -> W2Record:
    return W2Record(
        ssn=record.ssn, ein=record.ein, tax_year=record.tax_year,
        employer_name=record.employer_name, employee_name=record.employee_name,
        wages_box1=record.wages_box1, fed_income_tax_box2=record.fed_income_tax_box2,
        ss_wages_box3=record.ss_wages_box3, ss_tax_box4=record.ss_tax_box4,
        medicare_wages_box5=record.medicare_wages_box5, medicare_tax_box6=record.medicare_tax_box6,
        ss_tips_box7=record.ss_tips_box7, allocated_tips_box8=record.allocated_tips_box8,
        dependent_care_box10=record.dependent_care_box10, nonqualified_plans_box11=record.nonqualified_plans_box11,
        box13_retirement_plan=record.box13_retirement_plan,
        box12=list(record.box12), state_rows=list(record.state_rows),
    )


ARMS = {
    "hosted_prebuilt": SimulatedBackend("hosted_prebuilt", HOSTED_PREBUILT),
    "vlm_structured": SimulatedBackend("vlm_structured", VLM_STRUCTURED),
    "template_ocr": SimulatedBackend("template_ocr", TEMPLATE_OCR),
}
