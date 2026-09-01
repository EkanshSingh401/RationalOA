import csv
import io
import json

import pytest

from w2.chase import DocState
from w2.output import (
    ExportRecord,
    NotSignedOff,
    UnresolvedCriticalFindings,
    export_csv,
    export_json,
)
from w2.rules import expected_medicare_tax_cents, expected_ss_tax_cents
from w2.constants import ty
from w2.schema import Field, W2Record


def f(value, confidence=1.0, source="test"):
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def make_clean_record(tax_year=2024):
    consts = ty(tax_year)
    box1 = box5 = box3 = 100_000_00
    box7 = 0
    return W2Record(
        ssn=f("123-45-6789"),
        ein=f("12-3456789"),
        tax_year=f(tax_year),
        employer_name=f("Acme Corp"),
        employee_name=f("Jane Doe"),
        wages_box1=f(box1),
        fed_income_tax_box2=f(15_000_00),
        ss_wages_box3=f(box3),
        ss_tax_box4=f(expected_ss_tax_cents(box3, box7, consts)),
        medicare_wages_box5=f(box5),
        medicare_tax_box6=f(expected_medicare_tax_cents(box5, consts)),
        ss_tips_box7=f(box7),
        allocated_tips_box8=f(0),
        dependent_care_box10=f(0),
        nonqualified_plans_box11=f(0),
        box13_retirement_plan=f(False),
        box12=[],
        state_rows=[],
    )


def make_doc(doc_id="DOC-1", state=DocState.SIGNED_OFF, record=None):
    return ExportRecord(doc_id=doc_id, state=state, record=record or make_clean_record())


# --- the sign-off gate ---------------------------------------------------

def test_signed_off_record_exports_csv_and_json():
    doc = make_doc()
    csv_text = export_csv([doc])
    json_text = export_json([doc])

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "DOC-1"

    documents = json.loads(json_text)
    assert len(documents) == 1
    assert documents[0]["doc_id"] == "DOC-1"
    assert documents[0]["state"] == "SIGNED_OFF"


def test_ready_for_review_record_raises_for_both_formats():
    doc = make_doc(state=DocState.READY_FOR_REVIEW)

    with pytest.raises(NotSignedOff) as exc_info:
        export_csv([doc])
    assert "DOC-1" in str(exc_info.value)
    assert "READY_FOR_REVIEW" in str(exc_info.value)

    with pytest.raises(NotSignedOff):
        export_json([doc])


def test_non_signed_off_record_is_not_silently_dropped_from_a_batch():
    good = make_doc(doc_id="DOC-GOOD")
    bad = make_doc(doc_id="DOC-BAD", state=DocState.FLAGGED)

    # the whole batch must be refused, not silently reduced to the good ones
    with pytest.raises(NotSignedOff) as exc_info:
        export_csv([good, bad])
    assert "DOC-BAD" in str(exc_info.value)
    assert "FLAGGED" in str(exc_info.value)


# --- unresolved CRITICAL findings ----------------------------------------

def test_csv_refuses_signed_off_record_with_critical_finding():
    record = make_clean_record()
    record.ss_tax_box4 = f(1)  # now wildly wrong -> SS_TAX_MISMATCH, CRITICAL
    doc = make_doc(record=record)

    with pytest.raises(UnresolvedCriticalFindings) as exc_info:
        export_csv([doc])
    assert "DOC-1" in str(exc_info.value)
    assert "SS_TAX_MISMATCH" in str(exc_info.value)


def test_json_still_exports_but_flags_unresolved_critical_finding():
    record = make_clean_record()
    record.ss_tax_box4 = f(1)
    doc = make_doc(record=record)

    json_text = export_json([doc])  # must NOT raise
    documents = json.loads(json_text)

    assert len(documents) == 1
    assert documents[0]["has_unresolved_critical_findings"] is True
    rule_ids = {finding["rule_id"] for finding in documents[0]["findings"]}
    assert "SS_TAX_MISMATCH" in rule_ids


def test_clean_signed_off_record_has_no_flagged_findings():
    doc = make_doc()
    documents = json.loads(export_json([doc]))
    assert documents[0]["has_unresolved_critical_findings"] is False


# --- money format: dollars in CSV, cents in JSON --------------------------

def test_csv_renders_decimal_dollars():
    doc = make_doc()
    rows = list(csv.DictReader(io.StringIO(export_csv([doc]))))
    assert rows[0]["wages_box1"] == "100000.00"


def test_json_keeps_integer_cents():
    doc = make_doc()
    documents = json.loads(export_json([doc]))
    assert documents[0]["fields"]["wages_box1"]["value"] == 100_000_00
    assert isinstance(documents[0]["fields"]["wages_box1"]["value"], int)


# --- provenance in JSON --------------------------------------------------

def test_json_carries_full_field_provenance():
    doc = make_doc()
    documents = json.loads(export_json([doc]))
    ssn_field = documents[0]["fields"]["ssn"]
    assert set(ssn_field) == {"value", "confidence", "source", "bbox"}
    assert ssn_field["value"] == "123-45-6789"
    assert ssn_field["source"] == "test"
    assert ssn_field["confidence"] == 1.0
