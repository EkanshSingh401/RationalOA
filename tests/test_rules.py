import copy

import pytest

from w2 import generate, rules
from w2.constants import ty
from w2.rules import Finding, Rule, Severity, flagged_fields, validate
from w2.schema import Box12Entry, Field, StateRow, W2Record


def f(value, confidence=1.0, source="test"):
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def make_record(tax_year=2024, **overrides):
    consts = ty(tax_year)
    box1 = 100_000_00
    box5 = 100_000_00
    box3 = 100_000_00
    box7 = 0
    box4 = rules.expected_ss_tax_cents(box3, box7, consts)
    box6 = rules.expected_medicare_tax_cents(box5, consts)

    record = W2Record(
        ssn=f("123-45-6789"),
        ein=f("12-3456789"),
        tax_year=f(tax_year),
        employer_name=f("Acme Corp"),
        employee_name=f("Jane Doe"),
        wages_box1=f(box1),
        fed_income_tax_box2=f(15_000_00),
        ss_wages_box3=f(box3),
        ss_tax_box4=f(box4),
        medicare_wages_box5=f(box5),
        medicare_tax_box6=f(box6),
        ss_tips_box7=f(box7),
        allocated_tips_box8=f(0),
        dependent_care_box10=f(0),
        nonqualified_plans_box11=f(0),
        box13_retirement_plan=f(False),
        box12=[],
        state_rows=[StateRow(
            state=f("GA"),
            employer_state_id=f("123-45"),
            state_wages=f(90_000_00),
            state_income_tax=f(5_000_00),
        )],
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def rule_ids(findings):
    return {f.rule_id for f in findings}


def test_baseline_record_is_clean():
    assert validate(make_record()) == []


# --- arithmetic -------------------------------------------------------

def test_ss_tax_mismatch_fires():
    record = make_record(ss_tax_box4=f(1))
    assert "SS_TAX_MISMATCH" in rule_ids(validate(record))


def test_ss_tax_mismatch_tolerates_rounding():
    record = make_record()
    record.ss_tax_box4 = f(record.ss_tax_box4.value + 150)  # within $2 tolerance
    assert "SS_TAX_MISMATCH" not in rule_ids(validate(record))


def test_medicare_tax_mismatch_fires():
    record = make_record(medicare_tax_box6=f(1))
    assert "MEDICARE_TAX_MISMATCH" in rule_ids(validate(record))


def test_medicare_tax_mismatch_applies_additional_rate_above_threshold():
    consts = ty(2024)
    box5 = 25_000_000  # $250,000, above the $200,000 additional-Medicare threshold
    box6 = rules.expected_medicare_tax_cents(box5, consts)
    record = make_record(medicare_wages_box5=f(box5), medicare_tax_box6=f(box6),
                          wages_box1=f(box5), state_rows=[])
    assert validate(record) == []
    # a Box 6 that only applies the flat 1.45% (ignoring the 0.9% add-on) must fire
    flat_only = int(box5 * float(consts.medicare_rate))
    record2 = make_record(medicare_wages_box5=f(box5), medicare_tax_box6=f(flat_only),
                           wages_box1=f(box5), state_rows=[])
    assert "MEDICARE_TAX_MISMATCH" in rule_ids(validate(record2))


def test_ss_wage_base_exceeded_fires():
    consts = ty(2024)
    over = consts.ss_wage_base_cents + 1_000_00
    record = make_record(ss_wages_box3=f(over), ss_tax_box4=f(rules.expected_ss_tax_cents(over, 0, consts)))
    assert "SS_WAGE_BASE_EXCEEDED" in rule_ids(validate(record))


def test_box5_box1_unexplained_fires_without_box12():
    record = make_record(medicare_wages_box5=f(120_000_00))
    assert "BOX5_BOX1_UNEXPLAINED" in rule_ids(validate(record))


def test_box5_box1_explained_by_deferral_codes():
    record = make_record(
        wages_box1=f(80_000_00),
        medicare_wages_box5=f(100_000_00),
        ss_wages_box3=f(100_000_00),
        ss_tax_box4=f(rules.expected_ss_tax_cents(100_000_00, 0, ty(2024))),
        box12=[Box12Entry(code=f("D"), amount=f(20_000_00))],
    )
    assert "BOX5_BOX1_UNEXPLAINED" not in rule_ids(validate(record))


def test_roth_code_does_not_explain_box5_box1_gap():
    # AA (Roth) reduces nothing -- it must not be treated as an explaining deferral.
    record = make_record(
        wages_box1=f(80_000_00),
        medicare_wages_box5=f(100_000_00),
        ss_wages_box3=f(100_000_00),
        ss_tax_box4=f(rules.expected_ss_tax_cents(100_000_00, 0, ty(2024))),
        box12=[Box12Entry(code=f("AA"), amount=f(20_000_00))],
    )
    assert "BOX5_BOX1_UNEXPLAINED" in rule_ids(validate(record))


def test_box1_exceeds_box5_fires():
    record = make_record(wages_box1=f(150_000_00), medicare_wages_box5=f(100_000_00))
    assert "BOX1_EXCEEDS_BOX5" in rule_ids(validate(record))


def test_box1_equal_box5_does_not_fire():
    record = make_record(wages_box1=f(100_000_00), medicare_wages_box5=f(100_000_00))
    assert "BOX1_EXCEEDS_BOX5" not in rule_ids(validate(record))


def test_box3_exceeds_box1_unexplained_fires():
    record = make_record(wages_box1=f(50_000_00), ss_wages_box3=f(80_000_00),
                          ss_tax_box4=f(rules.expected_ss_tax_cents(80_000_00, 0, ty(2024))))
    assert "BOX3_EXCEEDS_BOX1_UNEXPLAINED" in rule_ids(validate(record))


def test_box3_exceeds_box1_explained_by_deferrals_does_not_fire():
    record = make_record(
        wages_box1=f(50_000_00),
        ss_wages_box3=f(80_000_00),
        ss_tax_box4=f(rules.expected_ss_tax_cents(80_000_00, 0, ty(2024))),
        box12=[Box12Entry(code=f("D"), amount=f(30_000_00))],
    )
    assert "BOX3_EXCEEDS_BOX1_UNEXPLAINED" not in rule_ids(validate(record))


def test_negative_amount_fires_for_scalar_field():
    record = make_record(fed_income_tax_box2=f(-1))
    assert "NEGATIVE_AMOUNT" in rule_ids(validate(record))


def test_negative_amount_fires_for_box12_entry():
    record = make_record(box12=[Box12Entry(code=f("D"), amount=f(-500))])
    findings = validate(record)
    assert "NEGATIVE_AMOUNT" in rule_ids(findings)
    neg = next(fnd for fnd in findings if fnd.rule_id == "NEGATIVE_AMOUNT")
    assert neg.fields == ("box12[D]_amount",)


def test_fed_tax_exceeds_wages_fires():
    record = make_record(fed_income_tax_box2=f(200_000_00))
    assert "FED_TAX_EXCEEDS_WAGES" in rule_ids(validate(record))


# --- identifiers --------------------------------------------------------

@pytest.mark.parametrize("ssn", ["123456789", "12-345-6789", "abc-de-fghi", ""])
def test_ssn_malformed_fires(ssn):
    record = make_record(ssn=f(ssn))
    assert "SSN_MALFORMED" in rule_ids(validate(record))


@pytest.mark.parametrize("ssn", ["000-45-6789", "666-45-6789", "900-45-6789", "999-45-6789"])
def test_ssn_invalid_area_fires(ssn):
    record = make_record(ssn=f(ssn))
    assert "SSN_INVALID_AREA" in rule_ids(validate(record))


def test_ssn_invalid_area_message_names_itin_for_9xx():
    record = make_record(ssn=f("912-45-6789"))
    findings = validate(record)
    finding = next(fnd for fnd in findings if fnd.rule_id == "SSN_INVALID_AREA")
    assert "ITIN" in finding.message


@pytest.mark.parametrize("ssn", ["123-00-6789", "123-45-0000"])
def test_ssn_invalid_group_serial_fires(ssn):
    record = make_record(ssn=f(ssn))
    assert "SSN_INVALID_GROUP_SERIAL" in rule_ids(validate(record))


@pytest.mark.parametrize("ein", ["123456789", "1-23456789", "12-345678"])
def test_ein_malformed_fires(ein):
    record = make_record(ein=f(ein))
    assert "EIN_MALFORMED" in rule_ids(validate(record))


# --- Box 12 / Box 13 -----------------------------------------------------

def test_box12_invalid_code_fires():
    record = make_record(box12=[Box12Entry(code=f("ZZ"), amount=f(1_000_00))])
    assert "BOX12_INVALID_CODE" in rule_ids(validate(record))


def test_box12_over_402g_fires():
    consts = ty(2024)
    over = consts.limit_402g_cents + consts.catch_up_50_cents + 1_000_00
    record = make_record(
        wages_box1=f(1),
        medicare_wages_box5=f(over + 1),
        ss_wages_box3=f(over + 1),
        ss_tax_box4=f(rules.expected_ss_tax_cents(over + 1, 0, consts)),
        box12=[Box12Entry(code=f("D"), amount=f(over))],
    )
    assert "BOX12_OVER_402G" in rule_ids(validate(record))


def test_box12_duplicate_code_fires_for_non_pl_code():
    record = make_record(box12=[
        Box12Entry(code=f("D"), amount=f(1_000_00)),
        Box12Entry(code=f("D"), amount=f(2_000_00)),
    ])
    assert "BOX12_DUPLICATE_CODE" in rule_ids(validate(record))


def test_box12_duplicate_code_allows_p_and_l():
    record = make_record(box12=[
        Box12Entry(code=f("P"), amount=f(1_000_00)),
        Box12Entry(code=f("P"), amount=f(2_000_00)),
        Box12Entry(code=f("L"), amount=f(500_00)),
        Box12Entry(code=f("L"), amount=f(600_00)),
    ])
    assert "BOX12_DUPLICATE_CODE" not in rule_ids(validate(record))


def test_box13_retirement_inconsistent_fires():
    record = make_record(
        wages_box1=f(80_000_00),
        medicare_wages_box5=f(100_000_00),
        ss_wages_box3=f(100_000_00),
        ss_tax_box4=f(rules.expected_ss_tax_cents(100_000_00, 0, ty(2024))),
        box12=[Box12Entry(code=f("D"), amount=f(20_000_00))],
        box13_retirement_plan=f(False),
    )
    assert "BOX13_RETIREMENT_INCONSISTENT" in rule_ids(validate(record))


def test_box13_retirement_consistent_does_not_fire():
    record = make_record(
        wages_box1=f(80_000_00),
        medicare_wages_box5=f(100_000_00),
        ss_wages_box3=f(100_000_00),
        ss_tax_box4=f(rules.expected_ss_tax_cents(100_000_00, 0, ty(2024))),
        box12=[Box12Entry(code=f("D"), amount=f(20_000_00))],
        box13_retirement_plan=f(True),
    )
    assert "BOX13_RETIREMENT_INCONSISTENT" not in rule_ids(validate(record))


# --- state ----------------------------------------------------------------

def test_no_tax_state_withholding_fires():
    record = make_record(state_rows=[StateRow(
        state=f("TX"), employer_state_id=f("1-2"),
        state_wages=f(90_000_00), state_income_tax=f(500_00),
    )])
    assert "NO_TAX_STATE_WITHHOLDING" in rule_ids(validate(record))


def test_state_wages_out_of_band_fires():
    record = make_record(state_rows=[StateRow(
        state=f("GA"), employer_state_id=f("1-2"),
        state_wages=f(1_000_00), state_income_tax=f(10_00),
    )])
    assert "STATE_WAGES_OUT_OF_BAND" in rule_ids(validate(record))


def test_state_tax_implausible_fires():
    record = make_record(state_rows=[StateRow(
        state=f("GA"), employer_state_id=f("1-2"),
        state_wages=f(90_000_00), state_income_tax=f(50_000_00),
    )])
    assert "STATE_TAX_IMPLAUSIBLE" in rule_ids(validate(record))


# --- engine behavior --------------------------------------------------

def test_unknown_tax_year_reports_finding_not_crash():
    record = make_record()
    record.tax_year = f(1999)
    findings = validate(record)
    assert rule_ids(findings) == {"UNKNOWN_TAX_YEAR"}


def test_rule_that_raises_is_caught_and_reported(monkeypatch):
    def boom(record, tax_year):
        raise ValueError("boom")

    broken_rule = Rule("BROKEN_RULE", Severity.CRITICAL, "raises on purpose", boom)
    patched = rules.RULES + [broken_rule]
    monkeypatch.setattr(rules, "RULES", patched)

    findings = validate(make_record())
    assert "BROKEN_RULE" in rule_ids(findings)
    broken = next(fnd for fnd in findings if fnd.rule_id == "BROKEN_RULE")
    assert broken.severity == Severity.CRITICAL
    assert "boom" in broken.message


def test_flagged_fields_unions_across_findings():
    findings = [
        Finding("A", Severity.CRITICAL, "x", ("ssn",)),
        Finding("B", Severity.WARN, "y", ("ein", "ssn")),
    ]
    assert flagged_fields(findings) == {"ssn", "ein"}


# --- generator: acceptance test + pinned invariant -------------------

def test_generator_produces_zero_findings_across_3000_clean_documents():
    findings_by_id = {}
    for i, record in enumerate(generate.generate_records(3000, seed=42)):
        findings = validate(record)
        if findings:
            findings_by_id.setdefault(findings[0].rule_id, []).append((i, findings))

    if findings_by_id:
        summary = {rule_id: len(hits) for rule_id, hits in findings_by_id.items()}
        first_rule = next(iter(findings_by_id))
        i, findings = findings_by_id[first_rule][0]
        raise AssertionError(
            f"generator produced findings on 'clean' data: {summary}; "
            f"first offender is record #{i}: {findings}"
        )


def test_generator_never_produces_box1_greater_than_box5():
    """Pins the mirror image of the negative-Box-1 bug: Box 1 must never
    exceed Box 5, since Box 5 is Box 1 plus deferrals."""
    for record in generate.generate_records(3000, seed=99):
        assert record.wages_box1.value <= record.medicare_wages_box5.value


def test_generator_never_produces_negative_box1():
    for record in generate.generate_records(3000, seed=7):
        assert record.wages_box1.value >= 0
