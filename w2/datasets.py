"""Load singhsays/fake-w2-us-tax-form-dataset from Hugging Face and adapt
its gt_parse records to W2Record.

Each row's "ground_truth" column is a JSON string shaped like
{"gt_parse": {<box_* fields>}}. Money fields arrive as JSON floats
(dollars, always <= 2 decimal places), not strings. A handful of fields
(Box 9, empty Box 12 code slots, unchecked Box 13 checkboxes) use the
literal string "None" as a not-present sentinel instead of JSON null --
money_to_cents() and from_gt_parse() both treat that sentinel as absent.
"""

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterator, Optional, Tuple

from datasets import load_dataset

from w2.schema import Box12Entry, Field, StateRow, W2Record

DATASET_NAME = "singhsays/fake-w2-us-tax-form-dataset"
SPLITS = ("train", "validation", "test")

# Box 12 codes for elective deferrals: excluded from Box 1 taxable wages
# but still counted in Box 3 / Box 5 SS & Medicare wages (IRS W-2 inst.).
DEFERRAL_BOX12_CODES = frozenset({"D", "E", "F", "G", "H", "S", "AA", "BB", "EE"})

_NONE_SENTINEL = "none"


def _is_none(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == _NONE_SENTINEL


def money_to_cents(value: Any) -> Optional[int]:
    """Parse a dollar amount (JSON float/int or numeric string) to integer
    cents. Returns None for the dataset's "None" sentinel."""
    if value is None or _is_none(value):
        return None
    try:
        return int((Decimal(str(value)) * 100).to_integral_value())
    except InvalidOperation:
        return None


def iter_raw_records(splits: Tuple[str, ...] = SPLITS) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
    """Yield (split, index, gt_parse) for every record in the dataset."""
    ds = load_dataset(DATASET_NAME)
    for split in splits:
        rows = ds[split].remove_columns(["image"])
        for i, row in enumerate(rows):
            yield split, i, json.loads(row["ground_truth"])["gt_parse"]


def _f(value: Any, confidence: float = 1.0, source: str = "gt_parse") -> Field:
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def _money_field(record: Dict[str, Any], key: str, source: str = "gt_parse") -> Optional[Field]:
    cents = money_to_cents(record[key])
    if cents is None:
        return None
    return Field(value=cents, confidence=1.0, source=source, bbox=None)


def _checkbox_field(record: Dict[str, Any], key: str, source: str = "gt_parse") -> Field:
    """Box 13 checkboxes: "x" means checked, the "None" sentinel means blank."""
    raw = record[key]
    checked = not _is_none(raw) and str(raw).strip().lower() == "x"
    return Field(value=checked, confidence=1.0, source=source, bbox=None)


def from_gt_parse(record: Dict[str, Any], tax_year: int, source: str = "gt_parse") -> W2Record:
    """Adapt one gt_parse dict to a W2Record.

    The dataset carries no tax-year field at all, so tax_year must be
    supplied by the caller. It's stored at confidence=0.0/source="assumed"
    to mark it as caller-supplied rather than extracted from the form.
    """
    box12 = []
    for slot in "abcd":
        code = record[f"box_12{slot}_code"]
        if _is_none(code):
            continue
        box12.append(Box12Entry(
            code=_f(code, source=source),
            amount=_money_field(record, f"box_12{slot}_value", source=source),
        ))

    state_rows = []
    for i in (1, 2):
        state = record[f"box_15_{i}_state"]
        if _is_none(state):
            continue
        state_rows.append(StateRow(
            state=_f(state, source=source),
            employer_state_id=_f(record[f"box_15_{i}_employee_state_id"], source=source),
            state_wages=_money_field(record, f"box_16_{i}_state_wages", source=source),
            state_income_tax=_money_field(record, f"box_17_{i}_state_income_tax", source=source),
        ))

    return W2Record(
        ssn=_f(record["box_a_employee_ssn"], source=source),
        ein=_f(record["box_b_employer_identification_number"], source=source),
        tax_year=_f(tax_year, confidence=0.0, source="assumed"),
        employer_name=_f(record["box_c_employer_name"], source=source),
        employee_name=_f(record["box_e_employee_name"], source=source),
        wages_box1=_money_field(record, "box_1_wages", source=source),
        fed_income_tax_box2=_money_field(record, "box_2_federal_tax_withheld", source=source),
        ss_wages_box3=_money_field(record, "box_3_social_security_wages", source=source),
        ss_tax_box4=_money_field(record, "box_4_social_security_tax_withheld", source=source),
        medicare_wages_box5=_money_field(record, "box_5_medicare_wages", source=source),
        medicare_tax_box6=_money_field(record, "box_6_medicare_wages_tax_withheld", source=source),
        ss_tips_box7=_money_field(record, "box_7_social_security_tips", source=source),
        allocated_tips_box8=_money_field(record, "box_8_allocated_tips", source=source),
        dependent_care_box10=_money_field(record, "box_10_dependent_care_benefits", source=source),
        nonqualified_plans_box11=_money_field(record, "box_11_nonqualified_plans", source=source),
        box13_retirement_plan=_checkbox_field(record, "box_13_retirement_plan", source=source),
        box12=box12,
        state_rows=state_rows,
    )


def load_records(tax_year: int, splits: Tuple[str, ...] = SPLITS) -> Iterator[Tuple[str, int, W2Record]]:
    """Load and adapt every record in the dataset to (split, index, W2Record)."""
    for split, i, raw in iter_raw_records(splits):
        yield split, i, from_gt_parse(raw, tax_year=tax_year)
