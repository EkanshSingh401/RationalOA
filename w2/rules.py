"""Declarative W-2 validation rules.

Rules are data, not control flow: RULES is a flat list of Rule(id,
severity, description, fn). validate(record) looks up the record's
TaxYear constants once, then runs every rule against the record and
collects the findings. A rule whose fn raises is caught and turned into
a CRITICAL finding for that rule -- it never drops the document.

Every check function takes (record, tax_year) and returns a list of
(message, fields) violations, where fields is a tuple of natural-key
field names (box12[D]_amount, state[GA]_box17_tax, ...) -- never a
positional index -- so a review UI can point at the exact boxes.

Money comparisons tolerate small rounding: DERIVED_TAX_TOLERANCE_CENTS
($2.00) for rate-derived tax amounts (SS/Medicare tax), DEFAULT_TOLERANCE_CENTS
($1.00) everywhere else, per the README's tolerance spec.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Callable, List, Optional, Tuple

from w2.constants import TaxYear, ty
from w2.schema import W2Record

DEFAULT_TOLERANCE_CENTS = 100
DERIVED_TAX_TOLERANCE_CENTS = 200

# Box 12 codes whose amounts are pre-tax elective deferrals excluded from
# Box 1 but still counted in Box 3 / Box 5 (the Box5-Box1 and Box3-Box1
# reconciliation checks). F and H are pre-tax but not 402(g)-limited.
PRETAX_WAGE_DEFERRAL_CODES = frozenset({"D", "E", "F", "G", "H", "S"})

# Codes sharing the combined annual 402(g) elective-deferral limit --
# traditional AND Roth deferrals count against the same cap.
CODE_402G_LIMIT_CODES = frozenset({"D", "E", "G", "S", "AA", "BB"})

# Codes that represent participation in an employer retirement plan, for
# the Box 13 "Retirement plan" checkbox consistency check.
RETIREMENT_BOX12_CODES = frozenset({"D", "E", "F", "G", "H", "S", "AA", "BB", "EE"})

DUPLICATE_ALLOWED_CODES = frozenset({"P", "L"})

SCALAR_MONEY_FIELD_NAMES = (
    "wages_box1",
    "fed_income_tax_box2",
    "ss_wages_box3",
    "ss_tax_box4",
    "medicare_wages_box5",
    "medicare_tax_box6",
    "ss_tips_box7",
    "allocated_tips_box8",
    "dependent_care_box10",
    "nonqualified_plans_box11",
)

SSN_RE = re.compile(r"^(\d{3})-(\d{2})-(\d{4})$")
EIN_RE = re.compile(r"^\d{2}-\d{7}$")


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARN = "WARN"


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    message: str
    fields: Tuple[str, ...]


Violation = Tuple[str, Tuple[str, ...]]
CheckFn = Callable[[W2Record, TaxYear], List[Violation]]


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    description: str
    fn: CheckFn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cents(field) -> int:
    return field.value if field is not None and field.value is not None else 0


def _sum_box12(record: W2Record, codes) -> int:
    return sum(
        e.amount.value
        for e in record.box12
        if e.code.value in codes and e.amount is not None and e.amount.value is not None
    )


def _round_cents(amount: Decimal) -> int:
    return int(amount.to_integral_value(rounding=ROUND_HALF_UP))


def expected_ss_tax_cents(box3_cents: int, box7_cents: int, tax_year: TaxYear) -> int:
    return _round_cents(Decimal(box3_cents + box7_cents) * tax_year.ss_rate)


def expected_medicare_tax_cents(box5_cents: int, tax_year: TaxYear) -> int:
    base = Decimal(box5_cents) * tax_year.medicare_rate
    excess = max(0, box5_cents - tax_year.additional_medicare_threshold_cents)
    additional = Decimal(excess) * tax_year.additional_medicare_rate
    return _round_cents(base + additional)


def _max_catchup_cents(tax_year: TaxYear) -> int:
    catchups = [c for c in (tax_year.catch_up_50_cents, tax_year.secure2_catch_up_60_63_cents) if c]
    return max(catchups, default=0)


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

def check_ss_tax_mismatch(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box3 = _cents(record.ss_wages_box3)
    box4 = _cents(record.ss_tax_box4)
    box7 = _cents(record.ss_tips_box7)
    expected = expected_ss_tax_cents(box3, box7, tax_year)
    if abs(box4 - expected) > DERIVED_TAX_TOLERANCE_CENTS:
        return [(
            f"Box 4 (${box4/100:,.2f}) does not match {tax_year.ss_rate:.3%} x "
            f"(Box3+Box7) = ${expected/100:,.2f}.",
            ("ss_tax_box4", "ss_wages_box3", "ss_tips_box7"),
        )]
    return []


def check_medicare_tax_mismatch(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box5 = _cents(record.medicare_wages_box5)
    box6 = _cents(record.medicare_tax_box6)
    expected = expected_medicare_tax_cents(box5, tax_year)
    if abs(box6 - expected) > DERIVED_TAX_TOLERANCE_CENTS:
        return [(
            f"Box 6 (${box6/100:,.2f}) does not match {tax_year.medicare_rate:.3%} x "
            f"Box5 plus {tax_year.additional_medicare_rate:.2%} on Box5 over "
            f"${tax_year.additional_medicare_threshold_cents/100:,.2f} = ${expected/100:,.2f}.",
            ("medicare_tax_box6", "medicare_wages_box5"),
        )]
    return []


def check_ss_wage_base_exceeded(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box3 = _cents(record.ss_wages_box3)
    box7 = _cents(record.ss_tips_box7)
    total = box3 + box7
    if total > tax_year.ss_wage_base_cents + DEFAULT_TOLERANCE_CENTS:
        return [(
            f"Box 3 + Box 7 (${total/100:,.2f}) exceeds the {tax_year.year} SS wage "
            f"base (${tax_year.ss_wage_base_cents/100:,.2f}).",
            ("ss_wages_box3", "ss_tips_box7"),
        )]
    return []


def check_box5_box1_unexplained(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box1 = _cents(record.wages_box1)
    box5 = _cents(record.medicare_wages_box5)
    deferral_sum = _sum_box12(record, PRETAX_WAGE_DEFERRAL_CODES)
    diff = (box5 - box1) - deferral_sum
    if abs(diff) > DEFAULT_TOLERANCE_CENTS:
        return [(
            f"Box 5 - Box 1 (${(box5-box1)/100:,.2f}) does not match the sum of "
            f"pre-tax Box 12 deferral codes D/E/F/G/H/S (${deferral_sum/100:,.2f}).",
            ("medicare_wages_box5", "wages_box1"),
        )]
    return []


def check_box1_exceeds_box5(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box1 = _cents(record.wages_box1)
    box5 = _cents(record.medicare_wages_box5)
    if box1 > box5:
        return [(
            f"Box 1 (${box1/100:,.2f}) exceeds Box 5 (${box5/100:,.2f}); Box 5 is "
            f"Box 1 plus deferrals and can never be smaller.",
            ("wages_box1", "medicare_wages_box5"),
        )]
    return []


def check_box3_exceeds_box1_unexplained(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box1 = _cents(record.wages_box1)
    box3 = _cents(record.ss_wages_box3)
    excess = box3 - box1
    if excess <= 0:
        return []
    deferral_sum = _sum_box12(record, PRETAX_WAGE_DEFERRAL_CODES)
    unexplained = excess - deferral_sum
    if unexplained > DEFAULT_TOLERANCE_CENTS:
        return [(
            f"Box 3 (${box3/100:,.2f}) exceeds Box 1 (${box1/100:,.2f}) by "
            f"${excess/100:,.2f}, but pre-tax Box 12 deferrals only account for "
            f"${deferral_sum/100:,.2f}.",
            ("ss_wages_box3", "wages_box1"),
        )]
    return []


def check_negative_amount(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    violations: List[Violation] = []
    for name in SCALAR_MONEY_FIELD_NAMES:
        f = getattr(record, name)
        if f is not None and f.value is not None and f.value < 0:
            violations.append((f"{name} is negative (${f.value/100:,.2f}).", (name,)))
    for entry in record.box12:
        v = entry.amount.value if entry.amount is not None else None
        if v is not None and v < 0:
            key = f"box12[{entry.code.value}]_amount"
            violations.append((f"{key} is negative (${v/100:,.2f}).", (key,)))
    for row in record.state_rows:
        state = row.state.value
        for suffix, f in (("box16_wages", row.state_wages), ("box17_tax", row.state_income_tax)):
            v = f.value if f is not None else None
            if v is not None and v < 0:
                key = f"state[{state}]_{suffix}"
                violations.append((f"{key} is negative (${v/100:,.2f}).", (key,)))
    return violations


def check_fed_tax_exceeds_wages(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box1 = _cents(record.wages_box1)
    box2 = _cents(record.fed_income_tax_box2)
    if box2 > box1 + DEFAULT_TOLERANCE_CENTS:
        return [(
            f"Box 2 (${box2/100:,.2f}) exceeds Box 1 (${box1/100:,.2f}).",
            ("fed_income_tax_box2", "wages_box1"),
        )]
    return []


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

def check_ssn_malformed(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    ssn = record.ssn.value or ""
    if not SSN_RE.match(ssn):
        return [(f"SSN {ssn!r} is not in NNN-NN-NNNN format.", ("ssn",))]
    return []


def check_ssn_invalid_area(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    m = SSN_RE.match(record.ssn.value or "")
    if not m:
        return []  # SSN_MALFORMED already covers this
    area = m.group(1)
    area_n = int(area)
    if 900 <= area_n <= 999:
        return [(
            f"SSN area {area!r} falls in the 900-999 ITIN range and can never be a "
            f"valid SSN on a W-2.",
            ("ssn",),
        )]
    if area_n == 0 or area == "666":
        return [(f"SSN area {area!r} is not a valid SSA-issued area.", ("ssn",))]
    return []


def check_ssn_invalid_group_serial(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    m = SSN_RE.match(record.ssn.value or "")
    if not m:
        return []
    group, serial = m.group(2), m.group(3)
    if group == "00" or serial == "0000":
        return [(
            f"SSN group/serial {group}-{serial} is invalid (group must not be 00, "
            f"serial must not be 0000).",
            ("ssn",),
        )]
    return []


def check_ein_malformed(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    ein = record.ein.value or ""
    if not EIN_RE.match(ein):
        return [(f"EIN {ein!r} is not in NN-NNNNNNN format.", ("ein",))]
    return []


# ---------------------------------------------------------------------------
# Box 12 / Box 13
# ---------------------------------------------------------------------------

def check_box12_invalid_code(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    violations: List[Violation] = []
    for entry in record.box12:
        code = entry.code.value
        if code not in tax_year.valid_box12_codes:
            violations.append((
                f"Box 12 code {code!r} is not valid for tax year {tax_year.year}.",
                (f"box12[{code}]_amount",),
            ))
    return violations


def check_box12_over_402g(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    total = _sum_box12(record, CODE_402G_LIMIT_CODES)
    limit = tax_year.limit_402g_cents + _max_catchup_cents(tax_year)
    if total > limit + DEFAULT_TOLERANCE_CENTS:
        fields = tuple(
            f"box12[{e.code.value}]_amount" for e in record.box12 if e.code.value in CODE_402G_LIMIT_CODES
        )
        return [(
            f"Elective deferrals (${total/100:,.2f}) exceed the {tax_year.year} "
            f"402(g) limit plus the largest applicable catch-up (${limit/100:,.2f}).",
            fields,
        )]
    return []


def check_box12_duplicate_code(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    counts: dict = {}
    for entry in record.box12:
        counts.setdefault(entry.code.value, []).append(entry)
    violations: List[Violation] = []
    for code, entries in counts.items():
        if len(entries) > 1 and code not in DUPLICATE_ALLOWED_CODES:
            key = f"box12[{code}]_amount"
            violations.append((
                f"Box 12 code {code!r} appears {len(entries)} times; only codes P "
                f"and L may legitimately repeat.",
                (key,),
            ))
    return violations


def check_box13_retirement_inconsistent(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    retirement_codes = sorted({e.code.value for e in record.box12 if e.code.value in RETIREMENT_BOX12_CODES})
    checked = bool(record.box13_retirement_plan and record.box13_retirement_plan.value)
    if retirement_codes and not checked:
        return [(
            f"Box 12 reports retirement deferral code(s) {retirement_codes}, but the "
            f"Box 13 'Retirement plan' checkbox is not checked.",
            ("box13_retirement_plan",),
        )]
    return []


# ---------------------------------------------------------------------------
# state (Boxes 15-20)
# ---------------------------------------------------------------------------

def check_no_tax_state_withholding(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    violations: List[Violation] = []
    for row in record.state_rows:
        state = row.state.value
        tax = row.state_income_tax.value if row.state_income_tax is not None else None
        if state in tax_year.no_income_tax_states and tax and tax > DEFAULT_TOLERANCE_CENTS:
            key = f"state[{state}]_box17_tax"
            violations.append((
                f"{state} has no personal income tax, but {key} reports "
                f"${tax/100:,.2f} withheld.",
                (key,),
            ))
    return violations


def check_state_wages_out_of_band(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    box1 = _cents(record.wages_box1)
    if box1 <= 0 or not record.state_rows:
        return []
    total_state_wages = sum(
        row.state_wages.value for row in record.state_rows if row.state_wages is not None and row.state_wages.value is not None
    )
    low, high = Decimal("0.55") * box1, Decimal("1.75") * box1
    if not (low <= total_state_wages <= high):
        fields = tuple(f"state[{row.state.value}]_box16_wages" for row in record.state_rows)
        return [(
            f"Total Box 16 state wages (${total_state_wages/100:,.2f}) is outside "
            f"0.55x-1.75x of Box 1 (${box1/100:,.2f}).",
            fields,
        )]
    return []


def check_state_tax_implausible(record: W2Record, tax_year: TaxYear) -> List[Violation]:
    violations: List[Violation] = []
    for row in record.state_rows:
        wages = row.state_wages.value if row.state_wages is not None else None
        tax = row.state_income_tax.value if row.state_income_tax is not None else None
        if wages is None or tax is None:
            continue
        if tax > Decimal("0.20") * wages + DEFAULT_TOLERANCE_CENTS:
            state = row.state.value
            key = f"state[{state}]_box17_tax"
            violations.append((
                f"{key} (${tax/100:,.2f}) exceeds 20% of state[{state}]_box16_wages "
                f"(${wages/100:,.2f}).",
                (key,),
            ))
    return violations


# ---------------------------------------------------------------------------
# rule set + engine
# ---------------------------------------------------------------------------

RULES: List[Rule] = [
    Rule("SS_TAX_MISMATCH", Severity.CRITICAL, "Box 4 == 6.2% x (Box 3 + Box 7)", check_ss_tax_mismatch),
    Rule("MEDICARE_TAX_MISMATCH", Severity.CRITICAL,
         "Box 6 == 1.45% x Box 5, plus 0.9% on Box 5 above the additional-Medicare threshold",
         check_medicare_tax_mismatch),
    Rule("SS_WAGE_BASE_EXCEEDED", Severity.CRITICAL,
         "Box 3 + Box 7 <= that year's SS wage base", check_ss_wage_base_exceeded),
    Rule("BOX5_BOX1_UNEXPLAINED", Severity.ERROR,
         "(Box 5 - Box 1) == sum of Box 12 pre-tax deferral codes (D, E, F, G, H, S)",
         check_box5_box1_unexplained),
    Rule("BOX1_EXCEEDS_BOX5", Severity.CRITICAL, "Box 1 <= Box 5", check_box1_exceeds_box5),
    Rule("BOX3_EXCEEDS_BOX1_UNEXPLAINED", Severity.ERROR,
         "Box 3 above Box 1 must be explained by Box 12 deferral codes",
         check_box3_exceeds_box1_unexplained),
    Rule("NEGATIVE_AMOUNT", Severity.CRITICAL, "No money box is negative", check_negative_amount),
    Rule("FED_TAX_EXCEEDS_WAGES", Severity.CRITICAL, "Box 2 <= Box 1", check_fed_tax_exceeds_wages),
    Rule("SSN_MALFORMED", Severity.CRITICAL, "Nine digits, NNN-NN-NNNN", check_ssn_malformed),
    Rule("SSN_INVALID_AREA", Severity.CRITICAL,
         "Area not 000, not 666, not 900-999 (ITIN)", check_ssn_invalid_area),
    Rule("SSN_INVALID_GROUP_SERIAL", Severity.CRITICAL,
         "Group != 00, serial != 0000", check_ssn_invalid_group_serial),
    Rule("EIN_MALFORMED", Severity.CRITICAL, "NN-NNNNNNN", check_ein_malformed),
    Rule("BOX12_INVALID_CODE", Severity.ERROR,
         "Code valid for that tax year", check_box12_invalid_code),
    Rule("BOX12_OVER_402G", Severity.WARN,
         "Elective deferrals <= 402(g) limit plus the largest applicable catch-up",
         check_box12_over_402g),
    Rule("BOX12_DUPLICATE_CODE", Severity.WARN,
         "No repeated codes, except P and L", check_box12_duplicate_code),
    Rule("BOX13_RETIREMENT_INCONSISTENT", Severity.WARN,
         "Box 13 retirement checkbox set when Box 12 shows retirement deferrals",
         check_box13_retirement_inconsistent),
    Rule("NO_TAX_STATE_WITHHOLDING", Severity.ERROR,
         "No Box 17 withholding in no-income-tax states", check_no_tax_state_withholding),
    Rule("STATE_WAGES_OUT_OF_BAND", Severity.WARN,
         "Sum of Box 16 within 0.55x-1.75x of Box 1", check_state_wages_out_of_band),
    Rule("STATE_TAX_IMPLAUSIBLE", Severity.ERROR,
         "Box 17 <= 20% of Box 16 for that row", check_state_tax_implausible),
]

_RULES_BY_ID = {r.id: r for r in RULES}
assert len(_RULES_BY_ID) == len(RULES), "duplicate rule id in RULES"


def validate(record: W2Record) -> List[Finding]:
    try:
        tax_year = ty(record.tax_year.value)
    except KeyError as exc:
        return [Finding("UNKNOWN_TAX_YEAR", Severity.CRITICAL, str(exc), ("tax_year",))]

    findings: List[Finding] = []
    for rule in RULES:
        try:
            violations = rule.fn(record, tax_year)
        except Exception as exc:  # a rule must never drop the document
            findings.append(Finding(
                rule.id, Severity.CRITICAL,
                f"internal error evaluating rule {rule.id}: {exc!r}",
                (),
            ))
            continue
        for message, fields in violations:
            findings.append(Finding(rule.id, rule.severity, message, tuple(fields)))
    return findings


def flagged_fields(findings: List[Finding]) -> set:
    result: set = set()
    for f in findings:
        result.update(f.fields)
    return result
