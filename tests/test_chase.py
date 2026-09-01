from datetime import date, timedelta

import pytest

from w2.chase import (
    CHASE_CADENCE,
    DocState,
    Evidence,
    ExpectedW2,
    IllegalTransition,
    compose_chase,
    dedupe,
    due_for_chase,
    reconcile,
    rollforward_expectations,
    transition,
)
from w2.schema import Field, W2Record


def f(value, confidence=1.0, source="test"):
    return Field(value=value, confidence=confidence, source=source, bbox=None)


def make_w2(ssn="123-45-6789", ein="12-3456789", tax_year=2024, employer_name="Acme Corp", confidence=1.0):
    return W2Record(
        ssn=f(ssn, confidence=confidence),
        ein=f(ein, confidence=confidence),
        tax_year=f(tax_year, confidence=confidence),
        employer_name=f(employer_name, confidence=confidence),
        employee_name=f("Jane Doe", confidence=confidence),
        wages_box1=f(100_000_00, confidence=confidence),
        fed_income_tax_box2=f(15_000_00, confidence=confidence),
        ss_wages_box3=f(100_000_00, confidence=confidence),
        ss_tax_box4=f(6_200_00, confidence=confidence),
        medicare_wages_box5=f(100_000_00, confidence=confidence),
        medicare_tax_box6=f(1_450_00, confidence=confidence),
    )


def make_expected(**overrides):
    defaults = dict(
        client_id="client-1",
        tax_year=2024,
        ein="12-3456789",
        employer_name="Acme Corp",
        evidence=Evidence.PRIOR_YEAR,
    )
    defaults.update(overrides)
    return ExpectedW2(**defaults)


# --- state machine ------------------------------------------------------

def test_illegal_transition_raises():
    expected = make_expected()
    with pytest.raises(IllegalTransition):
        transition(expected, DocState.SIGNED_OFF)


def test_legal_transition_updates_state_and_history():
    expected = make_expected()
    today = date(2025, 2, 1)
    transition(expected, DocState.REQUESTED, note="first chase", today=today)
    assert expected.state == DocState.REQUESTED
    assert expected.history == [(today, DocState.REQUESTED, "first chase")]


def test_superseded_reachable_from_signed_off():
    expected = make_expected(state=DocState.SIGNED_OFF)
    transition(expected, DocState.SUPERSEDED, note="W-2c received")
    assert expected.state == DocState.SUPERSEDED


def test_superseded_is_terminal():
    expected = make_expected(state=DocState.SUPERSEDED)
    with pytest.raises(IllegalTransition):
        transition(expected, DocState.RECEIVED)


def test_abandoned_is_reversible_to_received():
    expected = make_expected(state=DocState.ABANDONED)
    transition(expected, DocState.RECEIVED, note="client found the job after all")
    assert expected.state == DocState.RECEIVED


# --- completeness denominator -------------------------------------------

def test_rollforward_dedupes_by_ein_and_reconcile_finds_missing():
    prior_year = [
        make_w2(ssn="111-11-1111", ein="11-1111111", tax_year=2023, employer_name="Employer A"),
        make_w2(ssn="111-11-1111", ein="22-2222222", tax_year=2023, employer_name="Employer B"),
        # a corrected W-2c from the same employer -- same EIN, should not double the denominator
        make_w2(ssn="111-11-1111", ein="11-1111111", tax_year=2023, employer_name="Employer A"),
    ]

    expectations = rollforward_expectations(prior_year, client_id="client-1", tax_year=2024)

    eins = {e.ein for e in expectations}
    assert eins == {"11-1111111", "22-2222222"}
    assert len(expectations) == 2  # deduped, not 3

    # only Employer A's W-2 actually arrives this year
    received = [make_w2(ssn="111-11-1111", ein="11-1111111", tax_year=2024, employer_name="Employer A")]

    result = reconcile(expectations, received)

    assert len(result.matched) == 1
    assert result.matched[0][0].ein == "11-1111111"
    assert len(result.missing) == 1
    assert result.missing[0].ein == "22-2222222"
    assert result.unexpected == []
    assert result.duplicates == []


def test_reconcile_flags_unexpected_new_employer():
    expectations = rollforward_expectations(
        [make_w2(ssn="111-11-1111", ein="11-1111111", tax_year=2023, employer_name="Employer A")],
        client_id="client-1",
        tax_year=2024,
    )
    received = [
        make_w2(ssn="111-11-1111", ein="11-1111111", tax_year=2024, employer_name="Employer A"),
        make_w2(ssn="111-11-1111", ein="99-9999999", tax_year=2024, employer_name="New Employer"),
    ]

    result = reconcile(expectations, received)

    assert len(result.matched) == 1
    assert result.missing == []
    assert len(result.unexpected) == 1
    assert result.unexpected[0].ein.value == "99-9999999"


# --- dedupe --------------------------------------------------------------

def test_dedupe_keeps_highest_confidence_copy():
    blurry = make_w2(confidence=0.55)
    clear = make_w2(confidence=0.97)
    re_upload = make_w2(confidence=0.80)

    kept, dropped = dedupe([blurry, clear, re_upload])

    assert kept == [clear]
    assert len(dropped) == 2
    assert all(r in dropped for r in (blurry, re_upload))


def test_dedupe_leaves_distinct_records_alone():
    a = make_w2(ein="11-1111111")
    b = make_w2(ein="22-2222222")

    kept, dropped = dedupe([a, b])

    assert len(kept) == 2
    assert a in kept and b in kept
    assert dropped == []


# --- chase cadence + messages --------------------------------------------

def test_chase_body_always_has_https_link_and_never_invites_attachment():
    expected = make_expected()
    url = "https://secure.example.com/upload/abc123"
    for attempt_idx in range(len(CHASE_CADENCE)):
        expected.chases_sent = attempt_idx
        msg = compose_chase(expected, upload_url=url, client_name="Sam")
        body_lower = msg.body.lower()
        assert url in msg.body
        # the only mention of "attach" must be the warning not to, never an invitation to
        assert "attach" in body_lower
        assert "don't" in body_lower and "attached" in body_lower
        assert "please attach" not in body_lower


def test_chase_rejects_non_https_upload_url():
    expected = make_expected()
    with pytest.raises(ValueError):
        compose_chase(expected, upload_url="http://insecure.example.com/upload", client_name="Sam")


def test_only_final_touch_requires_human_approval():
    expected = make_expected()
    url = "https://secure.example.com/upload/abc123"
    approvals = []
    for attempt_idx in range(len(CHASE_CADENCE)):
        expected.chases_sent = attempt_idx
        msg = compose_chase(expected, upload_url=url, client_name="Sam")
        approvals.append(msg.requires_human_approval)
    assert approvals == [False] * (len(CHASE_CADENCE) - 1) + [True]


def test_tone_escalates_friendly_direct_final():
    expected = make_expected()
    url = "https://secure.example.com/upload/abc123"
    tones = []
    for attempt_idx in range(len(CHASE_CADENCE)):
        expected.chases_sent = attempt_idx
        tones.append(compose_chase(expected, upload_url=url, client_name="Sam").tone)
    assert tones[0] == "friendly"
    assert tones[-1] == "final"
    assert all(t == "direct" for t in tones[1:-1])


def test_compose_chase_refuses_beyond_last_touch():
    expected = make_expected(chases_sent=len(CHASE_CADENCE))
    with pytest.raises(ValueError):
        compose_chase(expected, upload_url="https://secure.example.com/upload", client_name="Sam")


def test_cadence_does_not_fire_before_season_start():
    expected = make_expected()
    season_start = date(2025, 1, 15)
    assert due_for_chase(expected, today=season_start - timedelta(days=1), season_start=season_start) is False


def test_cadence_fires_on_season_start_for_first_touch():
    expected = make_expected()
    season_start = date(2025, 1, 15)
    assert due_for_chase(expected, today=season_start, season_start=season_start) is True


def test_cadence_respects_gap_before_next_touch():
    season_start = date(2025, 1, 15)
    expected = make_expected(chases_sent=1, last_chase=season_start)
    # gap before touch 2 is CHASE_CADENCE[1] = 5 days
    assert due_for_chase(expected, today=season_start + timedelta(days=4), season_start=season_start) is False
    assert due_for_chase(expected, today=season_start + timedelta(days=5), season_start=season_start) is True


def test_cadence_stops_after_last_touch():
    expected = make_expected(chases_sent=len(CHASE_CADENCE), last_chase=date(2025, 2, 1))
    far_future = date(2025, 6, 1)
    assert due_for_chase(expected, today=far_future, season_start=date(2025, 1, 15)) is False


def test_cadence_ignores_documents_not_awaiting():
    expected = make_expected(state=DocState.RECEIVED)
    season_start = date(2025, 1, 15)
    assert due_for_chase(expected, today=season_start, season_start=season_start) is False
