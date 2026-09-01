"""Tax-year-keyed constants for W-2 processing.

All dollar amounts are integer cents (never float). Rates are Decimal.
Never inline a threshold in pipeline code -- add/extend a TaxYear entry
here instead, and look it up via ty(year).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class TaxYear:
    year: int

    # Social Security (OASDI)
    ss_wage_base_cents: int
    ss_rate: Decimal
    medicare_rate: Decimal
    additional_medicare_rate: Decimal
    additional_medicare_threshold_cents: int

    # Elective deferral limits (401(k)/403(b)/etc.)
    limit_402g_cents: int
    catch_up_50_cents: int
    # SECURE 2.0 higher catch-up for ages 60-63. None where not yet in effect.
    secure2_catch_up_60_63_cents: Optional[int]

    # HSA
    hsa_self_only_cents: int
    hsa_family_cents: int
    hsa_catch_up_cents: int

    # Health FSA
    fsa_health_limit_cents: int

    # Valid W-2 Box 12 codes
    valid_box12_codes: FrozenSet[str]

    # States with no personal income tax on wages
    no_income_tax_states: FrozenSet[str]


_BOX12_CODES: FrozenSet[str] = frozenset({
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "V", "W", "Y", "Z",
    "AA", "BB", "DD", "EE", "FF", "GG", "HH",
})

_NO_INCOME_TAX_STATES: FrozenSet[str] = frozenset({
    "AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY",
})

TAX_YEARS = {
    2024: TaxYear(
        year=2024,
        ss_wage_base_cents=16_860_000,
        ss_rate=Decimal("0.062"),
        medicare_rate=Decimal("0.0145"),
        additional_medicare_rate=Decimal("0.009"),
        additional_medicare_threshold_cents=20_000_000,
        limit_402g_cents=2_300_000,
        catch_up_50_cents=750_000,
        secure2_catch_up_60_63_cents=None,
        hsa_self_only_cents=415_000,
        hsa_family_cents=830_000,
        hsa_catch_up_cents=100_000,
        fsa_health_limit_cents=320_000,
        valid_box12_codes=_BOX12_CODES,
        no_income_tax_states=_NO_INCOME_TAX_STATES,
    ),
    2025: TaxYear(
        year=2025,
        ss_wage_base_cents=17_610_000,
        ss_rate=Decimal("0.062"),
        medicare_rate=Decimal("0.0145"),
        additional_medicare_rate=Decimal("0.009"),
        additional_medicare_threshold_cents=20_000_000,
        limit_402g_cents=2_350_000,
        catch_up_50_cents=750_000,
        secure2_catch_up_60_63_cents=1_125_000,
        hsa_self_only_cents=430_000,
        hsa_family_cents=855_000,
        hsa_catch_up_cents=100_000,
        fsa_health_limit_cents=330_000,
        valid_box12_codes=_BOX12_CODES,
        no_income_tax_states=_NO_INCOME_TAX_STATES,
    ),
}


def ty(year: int) -> TaxYear:
    """Look up tax constants for a year. Raises KeyError for unknown years."""
    try:
        return TAX_YEARS[year]
    except KeyError:
        known = ", ".join(str(y) for y in sorted(TAX_YEARS))
        raise KeyError(
            f"Unknown tax year {year!r}. Known years: {known}."
        ) from None
