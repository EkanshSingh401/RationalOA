# W-2 automation pipeline

Ingestion → extraction → validation → human review, for a tax-prep firm
receiving W-2s from clients. Every document gets mandatory human sign-off.

**Status: rule engine, generator, and evaluation harness built and measured.**
This README reports what was built and what the measurements showed. All
numbers below are pulled from [FINDINGS.md](FINDINGS.md) or are reproducible
by running the scripts in this repo — none are restated from memory. See
FINDINGS.md for full detail and methodology behind each result.

The single most important finding: the public benchmark dataset this project
was built around is arithmetically fake in specific, measurable ways — Box 4
is 7.65% of Box 3, not the correct 6.2%, and there is no Social Security wage
base cap at all (F1). That single fact drove most of what follows: it's why a
payroll-consistent generator had to be built before the rule engine could be
tested against anything, and why the extraction-arm comparison runs against
that generator rather than the public corpus.

---

## Quickstart

```bash
git clone <this-repo> && cd RationalOA
pip install -r requirements.txt
pytest
```

56 tests, no network access required except the two commands below that load
the Hugging Face dataset.

```bash
# Experiment 0: is the public dataset's arithmetic internally consistent?
python3 scripts/check_dataset_consistency.py

# Rule engine fire rates across all 2,000 public-dataset records
python3 scripts/score_public_dataset.py

# Extraction-arm comparison (paired shadow eval, 3 simulated backends)
python3 scripts/run_eval.py [n_docs] [seed]      # default 2000 2024

# Seeded-error study: rule-engine coverage per field and corruption type
python3 scripts/seeded_error_study.py [n_docs] [seed]   # default 8000 777
```

The first two download the HF dataset on first run (`datasets` caches it
locally after that). The last two use only the local generator — no network
needed.

---

## The thesis

Every W-2 gets human sign-off by design. That constraint reshapes what's worth
building:

- **Straight-through-processing rate is meaningless here.** It's zero on
  purpose. What matters is reviewer-minutes per document and escapes per
  document — errors that survive a reviewer who was already looking at the
  page.
- **Extraction is commodity.** Prebuilt document-AI models already cover W-2.
  Rebuilding that is a poor use of early engineering time.
- **The validation layer is the defensible work.** A W-2 is an
  over-determined system — Box 4 follows from Box 3, Box 6 from Box 5, Box 5
  minus Box 1 from Box 12 deferral codes. That arithmetic gives a correctness
  signal on *unlabeled production data* with no ground truth required, which
  is what you'd monitor in production and what points a reviewer's attention
  at the right box.

Build order followed from this: rule engine and evaluation harness first,
extraction models last and swappable behind one protocol.

---

## Findings

### F1 — The public dataset has systematically wrong FICA arithmetic

`singhsays/fake-w2-us-tax-form-dataset`, 2,000 records, is the primary corpus
this project was scoped around. Before writing any rule, Experiment 0 checked
whether its amounts satisfy real payroll arithmetic.

**The false start.** The naive check —
`Box4 − 0.062×(Box3+Box7)` and `Box5 − Box1 − ΣBox12 deferrals` — looked
conclusive: 0/2000 passed on both. That result was wrong. Formula B straddled
zero with a huge spread, consistent with unrelated random values. Formula A
did not: every record was short, never over, tight spread relative to the
mean — one-sidedness that clean is structure, not noise. The cause was a
defect in a *different* field: `Box7 ≡ Box3` and `Box8 ≡ Box5` in all 2,000
records (the generator copies tips rather than generating them), so
`Box3+Box7` double-counts wages and the residual is forced negative
regardless of whether Box 4 is correct. **The tell was the sign distribution,
not the pass rate.**

**What's actually true.** Re-tested on ratios, including `Box6/Box5`, which
no duplicated field touches: `Box4/Box3` = 0.076500 and `Box6/Box5` = 0.029000,
**zero variance**, across all 2,000 records. Box 4 = `round(0.0765×Box3)` and
Box 6 = `round(0.029×Box5)` hold to the cent in 1,998/2,000 records. The
coefficients aren't arbitrary: **0.0765 = 0.062 + 0.0145**, the combined
employee FICA rate applied where only the SS component belongs; **0.029 =
2×0.0145**, the employer+employee Medicare rate applied where only the
employee share belongs. Both are the same class of error — the wrong rate
from the right family — deterministic and reproducible, not noise.

**The more serious defect: no wage base cap.** Box 3 exceeds the 2024 SS wage
base ($168,600) in 753/2000 records (37.7%) and the 2025 base ($176,100) in
695/2000 (34.8%), with zero cap effect on Box 4 at either threshold. A wrong
rate is a recoverable constant factor — divide it out. **A missing cap
destroys information**: real Box 3 is `min(wages, base)`, which cannot be
inverted once capped. Those ~695 records aren't merely wrong, they're
structurally impossible.

**Consequence:** anyone fine-tuning an extraction model on this corpus is
teaching it a wrong tax world — that Box 4 is always 7.65% of Box 3 and Box 3
has no ceiling — and the model will reproduce that faithfully. This is the
concrete argument for keeping validation downstream of extraction regardless
of extractor quality, and it's why F1 prompted `BOX3_EXCEEDS_BOX1_UNEXPLAINED`
and the payroll-consistent generator both.

### F2 — The rule engine, validated against real data before being written

Every rule was checked against `scripts/score_public_dataset.py`'s fire rates
before shipping, and the acceptance target — **zero findings across 23,000
generated documents spanning 21 seeds** — was met after fixing two real bugs
the acceptance test and rule-design review surfaced (a generator bug that
produced degenerate near-zero Box 1 values for low earners, and the
`flat_fields()` collision below).

| Rule | Severity | Fire rate |
|---|---|---|
| `SS_TAX_MISMATCH` | CRITICAL | 2000/2000 (100.0%) |
| `MEDICARE_TAX_MISMATCH` | CRITICAL | 2000/2000 (100.0%) |
| `SS_WAGE_BASE_EXCEEDED` | CRITICAL | 1549/2000 (77.5%) |
| `BOX5_BOX1_UNEXPLAINED` | ERROR | 2000/2000 (100.0%) |
| `BOX1_EXCEEDS_BOX5` | CRITICAL | 1045/2000 (52.2%) |
| `BOX3_EXCEEDS_BOX1_UNEXPLAINED` | ERROR | 901/2000 (45.1%) |
| `NEGATIVE_AMOUNT` | CRITICAL | 0/2000 (0.0%) |
| `FED_TAX_EXCEEDS_WAGES` | CRITICAL | 0/2000 (0.0%) |
| `SSN_MALFORMED` / `SSN_INVALID_AREA` / `SSN_INVALID_GROUP_SERIAL` | CRITICAL | 0/2000 (0.0%) |
| `EIN_MALFORMED` | CRITICAL | 0/2000 (0.0%) |
| `BOX12_INVALID_CODE` | ERROR | 0/2000 (0.0%) |
| `BOX12_OVER_402G` | WARN | 0/2000 (0.0%) |
| `BOX12_DUPLICATE_CODE` | WARN | 611/2000 (30.6%) |
| `BOX13_RETIREMENT_INCONSISTENT` | WARN | 1238/2000 (61.9%) |
| `NO_TAX_STATE_WITHHOLDING` | ERROR | 701/2000 (35.0%) |
| `STATE_WAGES_OUT_OF_BAND` | WARN | 0/2000 (0.0%) |
| `STATE_TAX_IMPLAUSIBLE` | ERROR | 0/2000 (0.0%) |

`SS_WAGE_BASE_EXCEEDED` at 77.5% is not the ~35% Box 3 alone would suggest.
The rule checks Box 3 + Box 7 per spec, and Box 7 ≡ Box 3 in every record
(F1's tips-duplication artifact), so the checked total is effectively 2×
Box 3 — a rule-input artifact carried over from F1, not a new rule bug.
`BOX1_EXCEEDS_BOX5` at a near-coin-flip 52.2% is independent confirmation,
from the rule-firing side, of F1's conclusion that Box 1 and Box 5 are drawn
independently in this dataset. Every identifier/format rule sits at 0%,
because the generator (Faker) produces well-formed SSNs, EINs, and
plausible Box 12 amounts — this corpus validates the rule engine's
arithmetic layer and nothing else.

### F3 — Extraction arm comparison: a calibration inversion, and cost bought with escapes

2,000 documents from `w2.generate`, scored paired across three simulated
backends (`scripts/run_eval.py`, seed 2024):

| ARM | FIELD ACC | CRIT ACC | ESC/DOC | REV MIN | COST | P95 LAT | RC-AUC |
|---|---|---|---|---|---|---|---|
| `hosted_prebuilt` | 98.4% | 97.6% | 0.170 | 1.78m | $1.50 | 1217ms | 0.0003 |
| `vlm_structured` | 99.0% | 98.7% | 0.104 | 1.68m | $4.00 | 2628ms | 0.0022 |
| `template_ocr` | 95.5% | 93.5% | 0.469 | 2.25m | $0.05 | 301ms | 0.0020 |

ESC/DOC is escaped material errors per document (expected value under the
reviewer model, see F5). RC-AUC is area under the risk-coverage curve; lower
means confidence actually separates that arm's own right answers from wrong
ones.

`vlm_structured` has the best field accuracy (99.0%) but the worst RC-AUC
(0.0022, ~7x `hosted_prebuilt`'s 0.0003) — a strict calibration inversion,
confirmed programmatically. Its confidence barely separates correct from
incorrect fields, so a confidence threshold can't be trusted to prioritize
review. Under mandatory sign-off, confidence is what rations reviewer
attention, so selecting `vlm_structured` on field accuracy alone picks the
wrong arm.

The cost tradeoff, stated in dollars: `template_ocr` is **80x cheaper** than
`vlm_structured` ($0.05 vs. $4.00/doc) and buys that with **0.365 additional
escaped errors per document** (0.469 − 0.104). Per-error escape probability
is similar across all three arms (51–52%); the gap is almost entirely a
volume effect — `template_ocr` produces 1,814 material errors on this corpus
against `vlm_structured`'s 400, not a worse per-error catch rate.

### F4 — Seeded-error study: what the rule engine actually catches

8,000 documents, one injected error each, across Box 1–6 and SSN/EIN
(`scripts/seeded_error_study.py`, seed 777). **Flag rate is measured
directly** — real output of `rules.validate` on real corrupted records.

| FIELD | N | FLAG RATE | ESCAPE RATE |
|---|---|---|---|
| wages_box1 | 905 | 96.0% | 20.0% |
| **fed_income_tax_box2** | 996 | **13.7%** | 25.5% |
| ss_wages_box3 | 948 | 96.7% | 20.0% |
| ss_tax_box4 | 953 | 95.2% | 20.0% |
| medicare_wages_box5 | 994 | 97.1% | 20.0% |
| medicare_tax_box6 | 945 | 92.6% | 20.0% |
| **ssn** | 1116 | 56.4% | 46.2% |
| ein | 1143 | 53.5% | 47.9% |

Flag rate, not escape rate, is the number to trust here — Box 2 shows why.
Its escape rate (25.5%) looks *better* than SSN's (46.2%), which would
suggest Box 2 is fine. Breaking it down by corruption type shows that's a
dollar-weighting artifact: decimal-shift Box 2 errors average $759,961 in
apparent size and get caught 44.2% of the time (they blow past Box 1 and
trip `FED_TAX_EXCEEDS_WAGES`), while dropped/OCR/transposition Box 2 errors —
still real $10–34k mistakes — are caught **0.0–0.9%** of the time. The rare,
huge, easily-caught errors dominate the dollar-weighted average and hide that
the ordinary Box 2 error survives review almost every time it happens.

**Ranked by flag rate, the three worst-covered fields are `fed_income_tax_box2`
(13.7%), `ein` (53.5%), `ssn` (56.4%)** — dramatically worse than every
arithmetic-linked money field (92.6–97.1%). This empirically confirms, for
the first time rather than by reasoning about it, both gaps this project
flagged as concerns from the start: Box 2 is invisible to arithmetic, and SSN
structural validation catches almost nothing.

### F5 — Two bugs in the measuring instrument

Both were in code that measures the system, not in the system being
measured — worth calling out on its own, because a bug in the instrument is
worse than a bug in the thing measured: it doesn't announce itself. A bad
rule fires visibly; a bad metric just quietly reports a plausible, wrong
number.

**1. `flat_fields()` Box 12 collision.** The schema flattened repeated Box 12
rows to `box12[<code>]_amount`, keyed on code alone. A record with three `E`
entries — common on the public corpus, 611/2,000 records duplicate a code —
collapsed to one dict entry, silently dropping two real amounts. This isn't
just a labeling bug: `flat_fields()` is what an eval harness diffs truth
against prediction on, so two real amounts would never enter the scoring
denominator, and a score computed against a shrunk denominator looks better
than it is. Fixed by numbering entries within a duplicated code, ordered by
amount descending (Box 12 slot position carries no meaning, so an extractor
returning the same rows in a different order must not score as an error).

**2. The escape-rate denominator.** The original `materiality_weighted_escape_rate`
normalized by total dollars across all material errors — a per-*error*
metric, not a per-*document* one. That choice normalized away error volume:
it reported 20.3–20.4% for all three F3 arms, making them look nearly
identical on escape risk. They are not — `template_ocr` produces 4.5x more
material errors per document than `vlm_structured` on the same corpus. Fixed
by making escapes-per-document the headline metric (F3's ESC/DOC column),
keeping the dollar-weighted view as a secondary diagnostic.

Neither bug was caught by eyeballing output that looked reasonable — both
were caught by asking what denominator a ratio actually used and checking it
against a hand-built case with a known answer.

---

## What's measured vs. what's assumed

- **Measured:** every fire rate in F2 and F3's field-accuracy/RC-AUC/cost/
  latency numbers (real output of `rules.validate` and the simulated
  backends' error models). Every flag rate in F4.
- **Assumed:** every escape-rate number anywhere in this document rests on
  two constants at the top of `w2/evaluate.py` —
  `REVIEWER_CATCH_RATE_FLAGGED` (0.80) and `REVIEWER_CATCH_RATE_UNFLAGGED`
  (0.20) — plus `REVIEWER_BASE_MINUTES` / `REVIEWER_MINUTES_PER_FLAGGED_FIELD`
  for reviewer-minutes. These encode automation bias (a reviewer facing a
  flagged field reads carefully; one facing an unflagged field in an
  otherwise-clean form doesn't) but are stated numbers, not measurements.
  `scripts/seeded_error_study.py` is the instrument built to eventually
  replace them with real reviewer data — it doesn't replace them yet, since
  F4's escape-rate column still assumes the same two constants rather than
  measuring an actual human.
- **Simulated:** the three extraction backends in `w2/backends.py` are
  documented simulations driven by explicit error models (digit
  transposition, OCR confusion, decimal shift, dropped/hallucinated rows),
  not real models or APIs — no real extractor exists yet. They implement
  `extract_from_truth(record, rng)`, which is not part of the real `Backend`
  protocol (a real backend has no ground truth to peek at). A real backend
  implements `extract(image)` only, and swaps in behind `Backend` with no
  other change to `evaluate.py` or `run_eval.py` — the call site is the only
  thing that changes.

---

## Assumptions

**Direction of flow.** A tax-prep/CPA firm whose clients *send* W-2s in for
1040 preparation, not a firm issuing W-2s (which would be an EFW2/BSO filing
system with no extraction at all). If wrong, the rule engine survives as a
pre-filing gate — it has no dependency on the extractors.

**Intake mix.** Roughly 10–40k W-2s per season in a compressed January–April
window: ~45% digital payroll-provider PDFs, ~30% scans, ~20% phone photos,
~5% unusable. The load-bearing decision is routing on whether a text layer
exists before choosing a model — the digital-PDF share extracts near-100% for
near-zero cost, and OCRing it anyway is a common way these systems waste
money and introduce errors that were never in the document.

**Completeness has no denominator, unlike a W-9.** With W-9s, AP has a vendor
list — the firm knows who owes a form. With W-2s there's no such list: if a
client sends two, nothing says whether there was a third job. Three sources,
descending reliability: (1) IRS Wage & Income transcript — true ground truth,
but not reliably complete until well after the filing deadline, so it's a
post-filing reconciliation trigger, not an in-season check; (2) prior-year
rollforward — cheap, available in January, the in-season workhorse, assumes
~85% returning clients; (3) secondary signals (a state return implying wages
not on file, a Box 12 code implying an unmentioned plan).

**Security is non-negotiable.** Every W-2 is an SSN plus a full wage record —
GLBA Safeguards Rule and IRS Pub 4557 apply. Follow-up always directs to a
secure upload link, never an email attachment; any third-party model API
touching document images needs zero-retention terms in writing first;
field-level provenance (the `Field` wrapper's `source`) exists so a bad model
version's blast radius is answerable after the fact.

---

## Known gaps, and what I'd build next

In priority order, driven by what F3/F4 actually measured rather than a
priori guessing:

1. **A Box 2 plausibility control.** F4 measured this as the single
   worst-covered field (13.7% flag rate) — no arithmetic constraint exists on
   Box 2 on a real W-2, so it falls back entirely on unaided human attention,
   which the seeded-error study now shows is close to a coin flip for
   anything short of a catastrophic error. Needs a plausibility band from an
   effective-rate model or a second independent extraction with a
   disagreement check.
2. **Name/SSN matching**, not just structural validation. F4 measured SSN at
   56.4% flag rate — better than Box 2 but still the second-worst field, and
   structural validation catches almost nothing by construction (change one
   digit of a valid SSN and you usually get another valid SSN). The real
   control is matching against the prior-year return, the client record, or
   IRS TIN matching.
3. **Real reviewer data for `REVIEWER_CATCH_RATE_*`.** Every escape-rate
   number in this document is conditional on two assumed constants (see
   "What's measured vs. assumed"). Running the seeded-error-study
   methodology against actual human reviewers, not the assumed constants,
   turns every escape-rate number here from a projection into a measurement.
4. **The four real extraction arms behind `Backend`.** F3 measured the
   *shape* of a calibration inversion using simulated error models; the real
   question — does a real hosted API or a real VLM actually produce that
   inversion — needs `extract(image)` implementations. The harness is
   already built to swap them in with no other change.
5. **`chase.py`** — the document state machine and follow-up cadence
   (`EXPECTED → REQUESTED → RECEIVED → EXTRACTED → FLAGGED →
   READY_FOR_REVIEW → SIGNED_OFF`, plus `SUPERSEDED`/`ABANDONED`) and the
   completeness reconciliation logic. Not started.
6. **W-2c corrections and 4-up copy-sheet segmentation.** Out of scope for
   v1; `SUPERSEDED` exists as a state but the parser doesn't.
7. **W-3/941 cross-document reconciliation.** Only relevant if the firm also
   runs payroll for clients — a different reading of "direction of flow"
   than the one this project assumes.
