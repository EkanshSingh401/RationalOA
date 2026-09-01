"""Payroll-consistent synthetic W2Record generator.

Derivation order (README "Secondary: the payroll-consistent generator"),
every box is derived from the one before it -- nothing is sampled
independently:

    gross
      - section 125 (health premiums, FSA, HSA)   -> reduces Boxes 1, 3, 5
      + group-term life over $50k (Box 12 C)       -> increases Boxes 1, 3, 5
      = Box 5 (Medicare wages)
      - traditional elective deferrals (Box 12 D/E/G)  -> reduces Box 1 only
      = Box 1
    Box 3 = min(Box 5, wage base);  Box 4 = 6.2% x (Box 3 + Box 7)
    Box 6 = 1.45% x Box 5 + 0.9% on Box 5 over $200k

Box 4 / Box 6 reuse rules.expected_ss_tax_cents / expected_medicare_tax_cents
so the generator and the rule engine can never drift apart on the tax
formula itself -- only on the things a generator can legitimately get
wrong (derivation order, caps, sign).

Roth deferrals (Box 12 AA) reduce nothing and are generated deliberately,
since they're a real source of Box5-Box1 reconciliation confusion. Every
elective-deferral dollar (traditional + Roth) is drawn from one shared
402(g) budget, since the two share the same annual limit. Traditional
deferrals are additionally capped below Box 5 so Box 1 can never go
negative -- the mirror image of Box1-exceeds-Box5 and just as easy to
introduce by accident.
"""

import random
from decimal import Decimal
from typing import Iterator, List, Optional, Tuple

from w2.constants import TAX_YEARS, ty
from w2.rules import expected_medicare_tax_cents, expected_ss_tax_cents
from w2.schema import Box12Entry, Field, StateRow, W2Record

TRADITIONAL_DEFERRAL_CODES = ("D", "E", "G")

STATES = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
)


def _field(value, confidence: float = 1.0, source: str = "synthetic") -> Field:
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def _random_ssn(rng: random.Random) -> str:
    area = rng.randint(1, 899)
    while area == 666:
        area = rng.randint(1, 899)
    group = rng.randint(1, 99)
    serial = rng.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def _random_ein(rng: random.Random) -> str:
    prefix = rng.randint(1, 99)
    suffix = rng.randint(1, 9_999_999)
    return f"{prefix:02d}-{suffix:07d}"


def _split_amount(rng: random.Random, total_cents: int, parts: int) -> List[int]:
    """Split total_cents into `parts` non-negative shares summing exactly to it."""
    if parts <= 1:
        return [total_cents]
    weights = [rng.random() + 0.1 for _ in range(parts)]
    weight_total = sum(weights)
    amounts = [int(total_cents * w / weight_total) for w in weights]
    amounts[-1] += total_cents - sum(amounts)
    return amounts


def _generate_box12(rng: random.Random, tax_year, available_wages_cents: int) -> Tuple[List[Tuple[str, int]], int]:
    """Return (entries, traditional_deferral_total_cents).

    Traditional (D/E/G) and Roth (AA) deferrals share one 402(g) budget,
    kept well under the year's limit + largest catch-up so a clean record
    never trips BOX12_OVER_402G. The budget is also capped at half of
    available_wages_cents so a low earner can't be assigned a near-max
    deferral that would force Box 1 down to a degenerate few cents --
    that's a generator bug (unrealistic payroll), not a rule bug, even
    though the negative-Box-1 backstop below would technically still
    hold. Group-term life (C) is independent of the deferral budget --
    it isn't an elective deferral at all.
    """
    catchups = [c for c in (tax_year.catch_up_50_cents, tax_year.secure2_catch_up_60_63_cents) if c]
    cap = min(tax_year.limit_402g_cents + max(catchups, default=0), available_wages_cents // 2)
    budget = int(cap * rng.uniform(0.0, 0.8))

    entries: List[Tuple[str, int]] = []
    traditional_total = 0

    if budget >= 100 and rng.random() < 0.75:
        codes = rng.sample(TRADITIONAL_DEFERRAL_CODES, k=rng.randint(1, 2))
        include_roth = rng.random() < 0.35
        shares = _split_amount(rng, budget, len(codes) + (1 if include_roth else 0))
        if include_roth:
            roth_amount = shares.pop()
            if roth_amount >= 100:
                entries.append(("AA", roth_amount))
        for code, amount in zip(codes, shares):
            if amount >= 100:
                entries.append((code, amount))
                traditional_total += amount

    if rng.random() < 0.10:
        entries.append(("C", rng.randint(10_00, 300_00)))

    return entries, traditional_total


def _generate_state_row(rng: random.Random, tax_year, box1_cents: int) -> StateRow:
    state = rng.choice(STATES)
    factor = Decimal(str(round(rng.uniform(0.70, 1.30), 4)))
    state_wages = int(Decimal(box1_cents) * factor)
    if state in tax_year.no_income_tax_states:
        state_tax = 0
    else:
        rate = Decimal(str(round(rng.uniform(0.01, 0.10), 4)))
        state_tax = int(Decimal(state_wages) * rate)
    return StateRow(
        state=_field(state),
        employer_state_id=_field(f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"),
        state_wages=_field(state_wages),
        state_income_tax=_field(state_tax),
    )


def generate_w2(rng: random.Random, tax_year_value: int) -> W2Record:
    tax_year = ty(tax_year_value)

    gross = rng.randint(20_000_00, 400_000_00)

    section125 = 0
    if rng.random() < 0.5:
        section125 = min(rng.randint(50_00, 700_00), gross // 4)

    box12_entries, traditional_deferral_total = _generate_box12(rng, tax_year, gross - section125)
    group_term_life = sum(amount for code, amount in box12_entries if code == "C")

    box5 = gross - section125 + group_term_life

    # Cap traditional deferrals below available wages so Box 1 can never
    # go negative -- deferrals must be capped at available wages or you
    # produce an impossible negative Box 1.
    max_traditional = max(box5 - 1, 0)
    if traditional_deferral_total > max_traditional:
        scale = Decimal(max_traditional) / Decimal(traditional_deferral_total)
        box12_entries = [
            (code, int(Decimal(amount) * scale) if code in TRADITIONAL_DEFERRAL_CODES else amount)
            for code, amount in box12_entries
        ]
        traditional_deferral_total = sum(
            amount for code, amount in box12_entries if code in TRADITIONAL_DEFERRAL_CODES
        )

    box1 = box5 - traditional_deferral_total

    box3 = min(box5, tax_year.ss_wage_base_cents)
    room = tax_year.ss_wage_base_cents - box3
    box7 = 0
    if room > 0 and rng.random() < 0.15:
        box7 = rng.randint(0, min(room, int(box3 * 0.15) + 1))

    box4 = expected_ss_tax_cents(box3, box7, tax_year)
    box6 = expected_medicare_tax_cents(box5, tax_year)

    box2 = int(box1 * rng.uniform(0.05, 0.28))

    box10 = rng.choice([0, 0, rng.randint(100_00, 500_00)])
    box11 = rng.choice([0, 0, rng.randint(100_00, 300_00)])

    retirement_codes = {"D", "E", "F", "G", "H", "S", "AA", "BB", "EE"}
    box13_retirement = any(code in retirement_codes for code, _ in box12_entries)

    box12 = [Box12Entry(code=_field(code), amount=_field(amount)) for code, amount in box12_entries]

    return W2Record(
        ssn=_field(_random_ssn(rng)),
        ein=_field(_random_ein(rng)),
        tax_year=_field(tax_year_value),
        employer_name=_field(f"Employer {rng.randint(1000, 9999)} Inc"),
        employee_name=_field(f"Employee {rng.randint(1000, 9999)}"),
        wages_box1=_field(box1),
        fed_income_tax_box2=_field(box2),
        ss_wages_box3=_field(box3),
        ss_tax_box4=_field(box4),
        medicare_wages_box5=_field(box5),
        medicare_tax_box6=_field(box6),
        ss_tips_box7=_field(box7),
        allocated_tips_box8=_field(0),
        dependent_care_box10=_field(box10),
        nonqualified_plans_box11=_field(box11),
        box13_retirement_plan=_field(box13_retirement),
        box12=box12,
        state_rows=[_generate_state_row(rng, tax_year, box1)],
    )


def generate_records(n: int, seed: int = 0) -> Iterator[W2Record]:
    rng = random.Random(seed)
    years = sorted(TAX_YEARS)
    for _ in range(n):
        yield generate_w2(rng, rng.choice(years))
