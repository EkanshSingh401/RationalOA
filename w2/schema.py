"""W-2 record schema.

Every extracted value is wrapped in a Field (value, confidence, source,
bbox) -- never a bare scalar. Repeated rows (Box 12 entries, state rows)
key on their natural key (code / state abbreviation), never position, so
a dropped row never shifts the identity of the rows around it.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Field:
    value: Any
    confidence: float
    source: str
    bbox: Optional[BBox] = None


@dataclass
class Box12Entry:
    code: Field
    amount: Field


@dataclass
class StateRow:
    state: Field
    employer_state_id: Field
    state_wages: Field
    state_income_tax: Field


@dataclass
class W2Record:
    ssn: Field
    ein: Field
    tax_year: Field
    employer_name: Field
    employee_name: Field
    wages_box1: Field
    fed_income_tax_box2: Field
    ss_wages_box3: Field
    ss_tax_box4: Field
    medicare_wages_box5: Field
    medicare_tax_box6: Field
    ss_tips_box7: Optional[Field] = None
    allocated_tips_box8: Optional[Field] = None
    dependent_care_box10: Optional[Field] = None
    nonqualified_plans_box11: Optional[Field] = None
    box13_retirement_plan: Optional[Field] = None
    box12: List[Box12Entry] = field(default_factory=list)
    state_rows: List[StateRow] = field(default_factory=list)

    def key(self) -> Tuple[Any, Any, Any]:
        """Dedupe key: (SSN, EIN, year)."""
        return (self.ssn.value, self.ein.value, self.tax_year.value)

    def flat_fields(self) -> Dict[str, Field]:
        """Flatten this record to name -> Field.

        Repeated rows use natural keys (box12[<code>]_amount,
        state[<abbr>]_box17_tax, ...) instead of position, so a dropped
        row never shifts another row's key.

        A Box 12 code can legitimately repeat (P, L), and some data
        sources duplicate codes that shouldn't (see BOX12_DUPLICATE_CODE).
        Either way, every entry must still get its own key or amounts
        silently disappear from anything scored off this dict. Within a
        duplicated code, entries are ordered by amount descending -- not
        source/slot order, since Box 12 slot position (12a-12d) carries
        no meaning and an extractor returning the same rows in a
        different order must not count as an error -- and numbered
        box12[<code>#1]_amount, box12[<code>#2]_amount, ... A code that
        appears once keeps the plain box12[<code>]_amount form.
        """
        flat: Dict[str, Field] = {
            "ssn": self.ssn,
            "ein": self.ein,
            "tax_year": self.tax_year,
            "employer_name": self.employer_name,
            "employee_name": self.employee_name,
            "wages_box1": self.wages_box1,
            "fed_income_tax_box2": self.fed_income_tax_box2,
            "ss_wages_box3": self.ss_wages_box3,
            "ss_tax_box4": self.ss_tax_box4,
            "medicare_wages_box5": self.medicare_wages_box5,
            "medicare_tax_box6": self.medicare_tax_box6,
        }

        optional_fields = {
            "ss_tips_box7": self.ss_tips_box7,
            "allocated_tips_box8": self.allocated_tips_box8,
            "dependent_care_box10": self.dependent_care_box10,
            "nonqualified_plans_box11": self.nonqualified_plans_box11,
            "box13_retirement_plan": self.box13_retirement_plan,
        }
        for name, f in optional_fields.items():
            if f is not None:
                flat[name] = f

        by_code: Dict[Any, List[Box12Entry]] = defaultdict(list)
        for entry in self.box12:
            by_code[entry.code.value].append(entry)

        for code, entries in by_code.items():
            if len(entries) == 1:
                flat[f"box12[{code}]_amount"] = entries[0].amount
                continue
            ordered = sorted(
                entries,
                key=lambda e: (e.amount.value is None, -(e.amount.value or 0)),
            )
            for i, entry in enumerate(ordered, start=1):
                flat[f"box12[{code}#{i}]_amount"] = entry.amount

        for row in self.state_rows:
            state = row.state.value
            flat[f"state[{state}]_employer_state_id"] = row.employer_state_id
            flat[f"state[{state}]_box16_wages"] = row.state_wages
            flat[f"state[{state}]_box17_tax"] = row.state_income_tax

        return flat


CRITICAL_FIELDS = frozenset({
    "ssn",
    "ein",
    "wages_box1",
    "fed_income_tax_box2",
    "ss_wages_box3",
    "ss_tax_box4",
    "medicare_wages_box5",
    "medicare_tax_box6",
})


def is_critical(field_name: str) -> bool:
    return field_name in CRITICAL_FIELDS
