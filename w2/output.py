"""Downstream output: the last stage before a W-2 leaves this system.

Two formats, both built from the same canonical W2Record:

  - export_csv():  one flat row per W-2, decimal-dollar columns, meant to
    be handed to whatever a reviewer or a downstream system opens directly.
  - export_json(): the full record, with every value still wrapped in its
    Field -- confidence, source, bbox -- plus the rule findings computed
    at export time. This is the audit trail; export_csv() is the
    convenience view.

A real integration targets the tax software's own import schema (Drake,
UltraTax, Lacerte, etc.), not this module's CSV/JSON shape directly.
That's a deliberate, narrow adapter written against these two functions'
output -- the reason the canonical W2Record/Field schema exists at all is
so that adapting to one more downstream format is a small translation
layer, not a rewrite of everything upstream of it.

--- The sign-off gate is mechanical, not a policy described in prose ---

Every other module in this project assumes a human will look at a
document before it goes anywhere. That assumption is only real if
something actually enforces it. This module is where it becomes a
property of the code instead of a sentence in a README: export_csv()
and export_json() both refuse, by raising, any record whose current
DocState (w2.chase) is not SIGNED_OFF -- never silently, never a partial
skip. The exception names the offending doc_id and its actual state.

A second, narrower check catches a different failure: a document a human
DID mark SIGNED_OFF but that still carries an unresolved CRITICAL rule
finding (rules.validate is re-run here, at export time, rather than
trusting whatever findings existed when sign-off happened -- a human can
approve in error, and this is the last place in the pipeline that can
still catch it before the data leaves). The two formats respond
differently on purpose:

  - export_csv() REFUSES (raises UnresolvedCriticalFindings). The CSV is
    a flat row headed straight for a downstream import with no room to
    carry a warning -- there's no column for "but check this first," so
    a critical defect must not leave as a normal-looking row.
  - export_json() does NOT refuse. It exports the record but flags it
    loudly: every finding is serialized in `findings`, and
    `has_unresolved_critical_findings` is set at the top level of that
    document's entry. JSON is the audit trail; hiding the exceptional
    case from it would defeat the one thing it's for. A human reviewing
    the export sees the flag; nothing downstream of the JSON is assumed
    to look at it automatically.

--- Money ---

CSV columns are decimal dollar strings ("1500.00") -- what a downstream
consumer expects to import. JSON keeps every amount as the canonical
integer cents, unmodified, inside its Field wrapper -- the JSON export is
meant to be read by something that already speaks this project's schema,
so there's no reason to leave cents-precision behind converting it.
"""

import csv
import io
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from w2.chase import DocState
from w2.rules import Finding, Severity, validate
from w2.schema import Field, W2Record

# Box 12 has four lines (a-d) and Boxes 15-20 have two state-row lines on
# the paper form -- the CSV mirrors that fixed layout. Entries beyond
# these caps are dropped from the CSV by design (documented, not a bug);
# export_json() carries every entry regardless of count.
MAX_BOX12_SLOTS = 4
MAX_STATE_SLOTS = 2
_BOX12_SLOT_LABELS = ("a", "b", "c", "d")

CSV_COLUMNS = (
    ["doc_id", "tax_year", "ssn", "ein", "employer_name", "employee_name",
     "wages_box1", "fed_income_tax_box2", "ss_wages_box3", "ss_tax_box4",
     "medicare_wages_box5", "medicare_tax_box6", "ss_tips_box7", "allocated_tips_box8",
     "dependent_care_box10", "nonqualified_plans_box11", "box13_retirement_plan"]
    + [f"box12{slot}_{part}" for slot in _BOX12_SLOT_LABELS[:MAX_BOX12_SLOTS] for part in ("code", "amount")]
    + [f"state{n}{suffix}" for n in range(1, MAX_STATE_SLOTS + 1)
       for suffix in ("", "_employer_id", "_wages", "_tax")]
)


class NotSignedOff(Exception):
    """Raised when export is attempted on a record whose DocState isn't SIGNED_OFF."""


class UnresolvedCriticalFindings(Exception):
    """Raised (CSV only) when a SIGNED_OFF record still has an unresolved CRITICAL finding."""


@dataclass(frozen=True)
class ExportRecord:
    """The exporter's unit of work: a document identity, its lifecycle
    state (w2.chase.DocState), and the canonical extracted record."""
    doc_id: str
    state: DocState
    record: W2Record


def _require_signed_off(doc: ExportRecord) -> None:
    if doc.state != DocState.SIGNED_OFF:
        raise NotSignedOff(
            f"doc {doc.doc_id!r} is not signed off (state={doc.state.value}); refusing to export"
        )


def _critical_findings(record: W2Record) -> List[Finding]:
    return [f for f in validate(record) if f.severity == Severity.CRITICAL]


def _dollars(cents: Optional[int]) -> str:
    if cents is None:
        return ""
    return str((Decimal(cents) / 100).quantize(Decimal("0.01")))


def _field_value(f: Optional[Field]) -> Any:
    return f.value if f is not None else None


def _csv_row(doc: ExportRecord) -> Dict[str, Any]:
    record = doc.record
    row: Dict[str, Any] = {
        "doc_id": doc.doc_id,
        "tax_year": record.tax_year.value,
        "ssn": record.ssn.value,
        "ein": record.ein.value,
        "employer_name": record.employer_name.value,
        "employee_name": record.employee_name.value,
        "wages_box1": _dollars(record.wages_box1.value),
        "fed_income_tax_box2": _dollars(record.fed_income_tax_box2.value),
        "ss_wages_box3": _dollars(record.ss_wages_box3.value),
        "ss_tax_box4": _dollars(record.ss_tax_box4.value),
        "medicare_wages_box5": _dollars(record.medicare_wages_box5.value),
        "medicare_tax_box6": _dollars(record.medicare_tax_box6.value),
        "ss_tips_box7": _dollars(_field_value(record.ss_tips_box7)),
        "allocated_tips_box8": _dollars(_field_value(record.allocated_tips_box8)),
        "dependent_care_box10": _dollars(_field_value(record.dependent_care_box10)),
        "nonqualified_plans_box11": _dollars(_field_value(record.nonqualified_plans_box11)),
        "box13_retirement_plan": (
            "" if record.box13_retirement_plan is None or record.box13_retirement_plan.value is None
            else bool(record.box13_retirement_plan.value)
        ),
    }

    for i, slot in enumerate(_BOX12_SLOT_LABELS[:MAX_BOX12_SLOTS]):
        if i < len(record.box12):
            entry = record.box12[i]
            row[f"box12{slot}_code"] = entry.code.value
            row[f"box12{slot}_amount"] = _dollars(entry.amount.value)
        else:
            row[f"box12{slot}_code"] = ""
            row[f"box12{slot}_amount"] = ""

    for i in range(MAX_STATE_SLOTS):
        n = i + 1
        if i < len(record.state_rows):
            sr = record.state_rows[i]
            row[f"state{n}"] = sr.state.value
            row[f"state{n}_employer_id"] = sr.employer_state_id.value
            row[f"state{n}_wages"] = _dollars(sr.state_wages.value)
            row[f"state{n}_tax"] = _dollars(sr.state_income_tax.value)
        else:
            row[f"state{n}"] = ""
            row[f"state{n}_employer_id"] = ""
            row[f"state{n}_wages"] = ""
            row[f"state{n}_tax"] = ""

    return row


def export_csv(docs: List[ExportRecord]) -> str:
    """One flat row per SIGNED_OFF W-2, decimal-dollar money columns.

    Raises NotSignedOff on the first record not in DocState.SIGNED_OFF,
    and UnresolvedCriticalFindings on the first SIGNED_OFF record that
    still has an unresolved CRITICAL rule finding. Either way, nothing is
    written for a batch containing a bad record -- validate-then-write,
    not write-then-discover.
    """
    rows = []
    for doc in docs:
        _require_signed_off(doc)
        critical = _critical_findings(doc.record)
        if critical:
            raise UnresolvedCriticalFindings(
                f"doc {doc.doc_id!r} is signed off but still has {len(critical)} unresolved "
                f"CRITICAL finding(s) ({[f.rule_id for f in critical]}); refusing to export to CSV"
            )
        rows.append(_csv_row(doc))

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _bbox_to_json(f: Field) -> Optional[dict]:
    return asdict(f.bbox) if f.bbox is not None else None


def _field_to_json(f: Field) -> dict:
    return {"value": f.value, "confidence": f.confidence, "source": f.source, "bbox": _bbox_to_json(f)}


def _finding_to_json(finding: Finding) -> dict:
    return {
        "rule_id": finding.rule_id,
        "severity": finding.severity.value,
        "message": finding.message,
        "fields": list(finding.fields),
    }


def export_json(docs: List[ExportRecord]) -> str:
    """Full field-level provenance (value, confidence, source, bbox) for
    every field, natural-keyed via W2Record.flat_fields(), plus every
    rule finding computed fresh at export time. Money stays integer
    cents. Raises NotSignedOff exactly like export_csv() -- that gate is
    unconditional regardless of format. Does NOT raise on unresolved
    CRITICAL findings; instead every document's entry carries
    `has_unresolved_critical_findings` and the full `findings` list. See
    the module docstring for why the two formats differ here.
    """
    documents = []
    for doc in docs:
        _require_signed_off(doc)
        findings = validate(doc.record)
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        documents.append({
            "doc_id": doc.doc_id,
            "state": doc.state.value,
            "fields": {name: _field_to_json(f) for name, f in doc.record.flat_fields().items()},
            "findings": [_finding_to_json(f) for f in findings],
            "has_unresolved_critical_findings": bool(critical),
        })
    return json.dumps(documents, indent=2)
