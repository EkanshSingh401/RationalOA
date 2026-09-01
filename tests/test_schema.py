import pytest

from w2.constants import ty
from w2.schema import Box12Entry, Field, W2Record


def f(value, confidence=0.99, source="ocr"):
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def make_record(box12_entries=None, state_rows=None):
    return W2Record(
        ssn=f("123-45-6789"),
        ein=f("12-3456789"),
        tax_year=f(2024),
        employer_name=f("Acme Corp"),
        employee_name=f("Jane Doe"),
        wages_box1=f(500_000),
        fed_income_tax_box2=f(50_000),
        ss_wages_box3=f(500_000),
        ss_tax_box4=f(31_000),
        medicare_wages_box5=f(500_000),
        medicare_tax_box6=f(7_250),
        box12=box12_entries or [],
        state_rows=state_rows or [],
    )


def test_unknown_year_raises():
    with pytest.raises(KeyError, match="2099"):
        ty(2099)


def test_flat_fields_uses_natural_keys():
    record = make_record(box12_entries=[
        Box12Entry(code=f("D"), amount=f(100_000)),
        Box12Entry(code=f("DD"), amount=f(200_000)),
    ])

    flat = record.flat_fields()

    assert flat["box12[D]_amount"].value == 100_000
    assert flat["box12[DD]_amount"].value == 200_000
    # keys are the natural code, never a positional index
    assert "box12_0_amount" not in flat
    assert "box12_1_amount" not in flat


def test_dropped_box12_row_does_not_shift_other_keys():
    full = [
        Box12Entry(code=f("A"), amount=f(1_000)),
        Box12Entry(code=f("D"), amount=f(100_000)),
        Box12Entry(code=f("DD"), amount=f(200_000)),
    ]
    dropped = [full[0], full[2]]  # the "D" row went missing on extraction

    record = make_record(box12_entries=dropped)
    flat = record.flat_fields()

    assert "box12[D]_amount" not in flat
    assert flat["box12[A]_amount"].value == 1_000
    assert flat["box12[DD]_amount"].value == 200_000


def test_duplicate_box12_code_gets_one_key_per_entry():
    record = make_record(box12_entries=[
        Box12Entry(code=f("E"), amount=f(300)),
        Box12Entry(code=f("E"), amount=f(100)),
        Box12Entry(code=f("E"), amount=f(200)),
    ])

    flat = record.flat_fields()

    assert set(flat) & {"box12[E#1]_amount", "box12[E#2]_amount", "box12[E#3]_amount"} == {
        "box12[E#1]_amount", "box12[E#2]_amount", "box12[E#3]_amount",
    }
    # sorted by amount descending, so no entry is silently dropped
    assert flat["box12[E#1]_amount"].value == 300
    assert flat["box12[E#2]_amount"].value == 200
    assert flat["box12[E#3]_amount"].value == 100
    assert "box12[E]_amount" not in flat


def test_duplicate_box12_code_ordering_is_independent_of_source_order():
    entries_a = [
        Box12Entry(code=f("E"), amount=f(300)),
        Box12Entry(code=f("E"), amount=f(100)),
        Box12Entry(code=f("E"), amount=f(200)),
    ]
    entries_b = [
        Box12Entry(code=f("E"), amount=f(100)),
        Box12Entry(code=f("E"), amount=f(200)),
        Box12Entry(code=f("E"), amount=f(300)),
    ]

    flat_a = make_record(box12_entries=entries_a).flat_fields()
    flat_b = make_record(box12_entries=entries_b).flat_fields()

    values_a = {k: v.value for k, v in flat_a.items() if k.startswith("box12[E")}
    values_b = {k: v.value for k, v in flat_b.items() if k.startswith("box12[E")}
    assert values_a == values_b == {
        "box12[E#1]_amount": 300,
        "box12[E#2]_amount": 200,
        "box12[E#3]_amount": 100,
    }


def test_single_box12_entry_has_no_ordinal():
    record = make_record(box12_entries=[Box12Entry(code=f("E"), amount=f(100))])
    flat = record.flat_fields()
    assert flat["box12[E]_amount"].value == 100
    assert "box12[E#1]_amount" not in flat
