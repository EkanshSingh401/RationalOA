"""The follow-up loop: W-2s that were never sent, not ones that arrived.

Everything in w2/rules.py assumes a document is already in hand and asks
"is this correct." This module asks the question upstream of that: "is
this the complete set of documents this client should have sent us at
all." Extraction and validation can't see a W-2 that never arrived.

--- The completeness denominator ---

This is the part that differs most from a W-9 pipeline, and it deserves
more care than the state machine below. With a W-9, the denominator is
known: accounts payable has a vendor list, so the firm knows exactly who
it owes a 1099 to, and "did we get everyone" is a simple set difference.

A W-2 client has no equivalent list. If a client sends two W-2s, nothing
in that fact tells you whether there was a third job. The absence of a
document is not observable from the documents themselves -- it has to be
inferred from something outside them. Three sources, in descending order
of reliability:

  1. Prior-year rollforward (IMPLEMENTED below, rollforward_expectations).
     Chase any EIN present last year and missing this year. Cheap,
     available in January, correct most of the time -- the in-season
     workhorse. Assumes ~85% returning clients; a new client has no
     history to roll forward, so no denominator exists for them at all
     (see "New clients" below).

  2. IRS Wage & Income transcript (NOT implemented here). True ground
     truth -- the IRS already has every W-2 and 1099 reported against a
     given SSN, obtained via a signed Form 8821 or 2848 through
     e-Services/TDS. The caveat that makes this unusable as the primary
     in-season signal: it is not reliably complete until well after the
     filing deadline. Employers have until January 31 to file with the
     SSA, and IRS-side processing lags further. By the time the
     transcript is complete, the return this client needed it for may
     already be filed. That makes it a POST-FILING reconciliation and
     amended-return trigger, not an in-season check -- assuming otherwise
     would be the single biggest available design error in this system.

  3. Secondary signals (NOT implemented here). A state return implying
     wages the firm doesn't have a federal W-2 for; an organizer
     questionnaire answer ("I worked two jobs this year"); a Box 12 code
     D on file implying a 401(k) plan tied to an employer whose W-2
     never arrived. Weaker and noisier than rollforward, but catches
     genuinely new employers a prior-year diff can't.

New clients have no denominator at all -- there is no prior year to roll
forward and, usually, no transcript authorization yet either. The honest
answer there is an explicit client attestation ("list every employer you
had in <year>"), not an inference from any of the three sources above.
Silently trusting whatever the new client happens to send is the failure
mode this module exists to prevent.

--- Everything else in this module ---

DocState / TRANSITIONS / transition(): the state machine an ExpectedW2
moves through. dedupe(): Copy B, C, and 2 of the same W-2 (or a clearer
re-upload) are the same document, not three. CHASE_CADENCE /
due_for_chase() / compose_chase(): the actual follow-up cadence and
message content, escalating to a human on the last touch rather than
nagging indefinitely.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from w2.schema import W2Record


class DocState(Enum):
    EXPECTED = "EXPECTED"
    REQUESTED = "REQUESTED"
    RECEIVED = "RECEIVED"
    EXTRACTED = "EXTRACTED"
    FLAGGED = "FLAGGED"
    INFO_REQUESTED = "INFO_REQUESTED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    SIGNED_OFF = "SIGNED_OFF"
    SUPERSEDED = "SUPERSEDED"
    ABANDONED = "ABANDONED"


# Allowed next states per current state. A same-state "transition" (e.g.
# logging another chase touch while still REQUESTED) is always legal and
# doesn't need to appear here -- see transition() below.
TRANSITIONS: Dict[DocState, frozenset] = {
    DocState.EXPECTED: frozenset({DocState.REQUESTED, DocState.RECEIVED, DocState.ABANDONED}),
    DocState.REQUESTED: frozenset({DocState.RECEIVED, DocState.ABANDONED}),
    DocState.RECEIVED: frozenset({DocState.EXTRACTED}),
    DocState.EXTRACTED: frozenset({DocState.FLAGGED, DocState.READY_FOR_REVIEW}),
    DocState.FLAGGED: frozenset({DocState.INFO_REQUESTED, DocState.READY_FOR_REVIEW}),
    DocState.INFO_REQUESTED: frozenset({DocState.RECEIVED, DocState.READY_FOR_REVIEW}),
    DocState.READY_FOR_REVIEW: frozenset({DocState.SIGNED_OFF, DocState.FLAGGED}),
    # SUPERSEDED is reachable from SIGNED_OFF: a W-2c or a clearer scan can
    # replace an already-approved record, so sign-off can't be modeled as
    # final.
    DocState.SIGNED_OFF: frozenset({DocState.SUPERSEDED}),
    DocState.SUPERSEDED: frozenset(),  # terminal
    # Reversible: ABANDONED means the client confirmed the job didn't
    # exist, and clients are sometimes wrong about that.
    DocState.ABANDONED: frozenset({DocState.RECEIVED}),
}


class IllegalTransition(Exception):
    pass


class Evidence(str, Enum):
    """Why we think this document exists at all."""
    PRIOR_YEAR = "prior_year"
    TRANSCRIPT = "transcript"
    ORGANIZER = "organizer"
    STATE_RETURN = "state_return"


@dataclass
class ExpectedW2:
    client_id: str
    tax_year: int
    ein: Optional[str]  # nullable -- sometimes we only know the employer by name
    employer_name: str
    evidence: Evidence
    state: DocState = DocState.EXPECTED
    chases_sent: int = 0
    last_chase: Optional[date] = None
    history: List[Tuple[date, DocState, str]] = field(default_factory=list)


def transition(expected: ExpectedW2, new_state: DocState, note: str = "", today: Optional[date] = None) -> None:
    """Move expected to new_state, or raise IllegalTransition. Appends to
    expected.history either way it succeeds -- history is append-only,
    this is the only function that should write to it."""
    today = today or date.today()
    current = expected.state
    if new_state != current and new_state not in TRANSITIONS.get(current, frozenset()):
        raise IllegalTransition(f"cannot move from {current.value} to {new_state.value}")
    expected.state = new_state
    expected.history.append((today, new_state, note))


# ---------------------------------------------------------------------------
# completeness denominator: source 1 (rollforward) + reconciliation
# ---------------------------------------------------------------------------

def rollforward_expectations(
    prior_year_records: List[W2Record], client_id: str, tax_year: int
) -> List[ExpectedW2]:
    """Build tax_year's denominator from prior_year_records -- the prior
    tax year's SIGNED_OFF W-2s for this client, deduped by EIN. (Filtering
    to "signed off" and "this client" is the caller's job: W2Record itself
    carries neither a client id nor a document state -- those live on
    ExpectedW2 and the chase pipeline, not on the extracted record.)

    Two W-2s from the same employer in one year (a mid-year raise
    triggering a corrected W-2, e.g.) collapse to one expectation --
    rollforward answers "which employers," not "how many documents."
    """
    seen_eins: Dict[str, W2Record] = {}
    for record in prior_year_records:
        ein = record.ein.value
        if ein not in seen_eins:
            seen_eins[ein] = record

    return [
        ExpectedW2(
            client_id=client_id,
            tax_year=tax_year,
            ein=ein,
            employer_name=record.employer_name.value,
            evidence=Evidence.PRIOR_YEAR,
        )
        for ein, record in seen_eins.items()
    ]


@dataclass
class Reconciliation:
    matched: List[Tuple[ExpectedW2, W2Record]]
    missing: List[ExpectedW2]     # chase these
    unexpected: List[W2Record]    # a new employer -- fine, but confirm nothing else was omitted
    duplicates: List[W2Record]    # re-uploads of a document already matched or unexpected


def reconcile(expected: List[ExpectedW2], received: List[W2Record]) -> Reconciliation:
    """Match this year's denominator against what actually arrived.

    `received` is deduped internally first (dedupe()) -- Copy B/C/2 and
    clearer re-uploads are the same document, not three separate ones,
    and shouldn't count as three separate employers on either side of
    the reconciliation.
    """
    kept, dropped = dedupe(received)

    by_ein: Dict[str, List[W2Record]] = defaultdict(list)
    for record in kept:
        by_ein[record.ein.value].append(record)

    expected_eins = {e.ein for e in expected if e.ein}

    matched: List[Tuple[ExpectedW2, W2Record]] = []
    missing: List[ExpectedW2] = []
    for e in expected:
        candidates = by_ein.get(e.ein, []) if e.ein else []
        if candidates:
            matched.append((e, candidates[0]))
        else:
            missing.append(e)

    unexpected = [r for r in kept if r.ein.value not in expected_eins]

    return Reconciliation(matched=matched, missing=missing, unexpected=unexpected, duplicates=dropped)


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------

def _mean_confidence(record: W2Record) -> float:
    fields = list(record.flat_fields().values())
    if not fields:
        return 0.0
    return statistics.mean(f.confidence for f in fields)


def dedupe(records: List[W2Record]) -> Tuple[List[W2Record], List[W2Record]]:
    """Collapse records on (SSN, EIN, tax year) -- Copy B, C, and 2 of a
    W-2 are the same document, and clients re-upload clearer photos of
    the same document. Keeps the copy with the highest mean field
    confidence per group. Returns (kept, dropped)."""
    groups: Dict[tuple, List[W2Record]] = defaultdict(list)
    for record in records:
        groups[record.key()].append(record)

    kept: List[W2Record] = []
    dropped: List[W2Record] = []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        best = max(group, key=_mean_confidence)
        kept.append(best)
        dropped.extend(r for r in group if r is not best)

    return kept, dropped


# ---------------------------------------------------------------------------
# chase cadence + messages
# ---------------------------------------------------------------------------

# Gap in days before each successive touch, measured from the previous
# contact (touch 1's "previous contact" is season_start). Four touches
# across ~4 weeks (0+5+9+14=28 days), not weekly -- response rates
# collapse and clients start filtering the sender on a weekly cadence.
CHASE_CADENCE = [0, 5, 9, 14]

_AWAITING_STATES = (DocState.EXPECTED, DocState.REQUESTED)


def due_for_chase(expected: ExpectedW2, today: date, season_start: date) -> bool:
    """Whether expected is due for its next chase touch today. False
    before season_start (cadence doesn't fire early), false once every
    scheduled touch has been sent (stop and escalate to the human
    relationship owner instead of continuing), false if the document
    isn't in an awaiting-document state any more."""
    if expected.state not in _AWAITING_STATES:
        return False
    if today < season_start:
        return False
    if expected.chases_sent >= len(CHASE_CADENCE):
        return False
    reference = expected.last_chase if expected.last_chase is not None else season_start
    gap_days = CHASE_CADENCE[expected.chases_sent]
    return today >= reference + timedelta(days=gap_days)


@dataclass(frozen=True)
class ChaseMessage:
    subject: str
    body: str
    tone: str
    attempt: int
    requires_human_approval: bool


_NO_ATTACHMENT_LINE = (
    "For your security, please don't reply to this email with your W-2 attached -- "
    "email is not a secure way to send an SSN and wage information. Use the secure "
    "upload link above instead."
)


def compose_chase(expected: ExpectedW2, upload_url: str, client_name: str) -> ChaseMessage:
    """Compose the next chase touch for expected. Always directs to
    upload_url (must be https) and always tells the client not to send
    the document as an email attachment -- a W-2 in an email body is an
    SSN plus a full wage record in plaintext, which runs straight into
    the GLBA Safeguards Rule and IRS Pub 4557."""
    if not upload_url.startswith("https://"):
        raise ValueError(f"upload_url must be a secure (https) link, got {upload_url!r}")

    attempt = expected.chases_sent + 1
    if attempt > len(CHASE_CADENCE):
        raise ValueError(
            "no more scheduled touches for this document -- escalate to the "
            "relationship owner instead of composing another chase"
        )
    is_final = attempt == len(CHASE_CADENCE)
    tone = "friendly" if attempt == 1 else ("final" if is_final else "direct")

    subject, opening, closing = {
        "friendly": (
            f"Quick reminder: your {expected.tax_year} W-2 from {expected.employer_name}",
            f"Hi {client_name},\n\n"
            f"Just a quick reminder that we're still waiting on your {expected.tax_year} "
            f"W-2 from {expected.employer_name} to finish your return.",
            "Whenever you get a chance, please upload it here:",
        ),
        "direct": (
            f"Still need your {expected.tax_year} W-2 from {expected.employer_name}",
            f"Hi {client_name},\n\n"
            f"We still haven't received your {expected.tax_year} W-2 from "
            f"{expected.employer_name}, and we need it to move your return forward.",
            "Please upload it as soon as you can here:",
        ),
        "final": (
            f"Final reminder: {expected.tax_year} W-2 from {expected.employer_name} still needed",
            f"Hi {client_name},\n\n"
            f"This is our final automated reminder about your outstanding {expected.tax_year} "
            f"W-2 from {expected.employer_name}. If we don't hear back, a member of our team "
            f"will reach out directly.",
            "Please upload it now here:",
        ),
    }[tone]

    body = f"{opening}\n\n{closing} {upload_url}\n\n{_NO_ATTACHMENT_LINE}\n\nThanks,\nYour tax team"

    return ChaseMessage(
        subject=subject,
        body=body,
        tone=tone,
        attempt=attempt,
        requires_human_approval=is_final,
    )
