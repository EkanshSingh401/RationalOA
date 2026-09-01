# W-2 automation pipeline

Ingestion → extraction → validation → human review → downstream output, for a tax
preparation firm receiving W-2s from clients.

This README is the spec.
Everything below is a decision I made and can defend; "Open questions" at the
bottom lists what I would change my mind about given real data.

---

## The thesis

Every W-2 gets human sign-off by design. That single constraint reshapes the
project:

- **Straight-through-processing rate is meaningless here.** It is zero on purpose.
  What matters is reviewer-minutes per document and materiality-weighted escape
  rate — errors that survive a reviewer who was already looking at the document.
- **Extraction is commodity.** Azure Document Intelligence, Google Document AI, and
  AWS Textract all ship prebuilt US tax form models covering W-2. Rebuilding that
  is a poor use of early engineering time.
- **The validation layer is the defensible work.** A W-2 is an over-determined
  system: Box 4 is 6.2% of Box 3, Box 6 follows from Box 5, and Box 5 minus Box 1
  must be explained by Box 12 deferral codes. That arithmetic gives a correctness
  signal on *unlabeled production data* with no ground truth required — which is
  what you monitor on, and what points a reviewer's attention at the right box.

Build order follows: rule engine and evaluation harness first, extraction models
last and swappable.

---

## Data

### Primary: `singhsays/fake-w2-us-tax-form-dataset`

2,000 W-2 images with structured ground truth. 1,800 train / 100 validation / 100
test, ~310 MB. Synthetically generated names, SSNs, EINs and addresses, with real
city/state/zip combinations. Derived from the Kaggle *Fake W-2 (US Tax Form)
Dataset*.

Ground truth is Donut `gt_parse` JSON keyed by box:

```json
{"gt_parse": {
  "box_b_employer_identification_number": "47-5592725",
  "box_c_employer_name": "Bennett, Allen and Yang Inc",
  "box_a_employee_ssn": "412-88-2525",
  "box_e_employee_name": "Michele...", ...}}
```

This is a real labeled eval set on real rendered images, which is much better than
scoring against my own generator. It becomes the primary A/B corpus.

### Experiment 0: are the amounts internally consistent?

**Run this before anything else.** Load the ground truth, compute
`Box 4 − 0.062 × (Box 3 + Box 7)` and `Box 5 − Box 1 − Σ(Box 12 deferrals)`, and
plot the distributions.

My hypothesis is that this dataset sampled each box independently, so the FICA
arithmetic will **not** hold. Either result is useful:

- **If inconsistent** (expected): the rule engine will fire on nearly every
  document, which is a clean demonstration that it works — and a principled reason
  to keep a payroll-consistent generator for anything involving training or
  threshold tuning. Report the fraction of the public dataset that fails each rule.
  That number is a finding worth putting in the writeup.
- **If consistent**: the generator becomes far less necessary and I should say so
  and lean on the public data.

This one script converts "I used a Hugging Face dataset" into "I characterized a
Hugging Face dataset and found something about it," which is the difference the
grader is looking for.

### Secondary: the payroll-consistent generator

Still built, for the reasons above. It does **not** sample box values
independently — it simulates a payroll year and derives every box:

```
gross
  − section 125 (health premiums, FSA, HSA)   → reduces Boxes 1, 3, 5
  + group-term life over $50k (Box 12 C)      → increases Boxes 1, 3, 5
  = Box 5 (Medicare wages)
  − traditional elective deferrals (Box 12 D/E/G)  → reduces Box 1 only
  = Box 1
Box 3 = min(Box 5, wage base);  Box 4 = 6.2% × (Box 3 + Box 7)
Box 6 = 1.45% × Box 5 + 0.9% on Box 5 over $200k
```

Roth deferrals (Box 12 AA) reduce nothing — a real source of Box 5 − Box 1
confusion, so generate them deliberately. Deferrals must be capped at available
wages or you produce a negative Box 1, which is impossible on a real W-2.

Its jobs: (1) supply the cases the public set lacks, (2) act as a correctness test
for the rules — **zero findings on a few thousand clean generated documents**, and
(3) let seeded-error studies control the injected error exactly.

### What the public dataset does not cover

Worth stating plainly, because a 2k-image single-generator corpus is not a
production distribution:

| Missing | Why it matters |
|---|---|
| Multi-state rows (Boxes 15–20) | Dropping the second state row is a top real failure mode |
| Box 12 code variety | The Box 5 − Box 1 reconciliation rule is the highest-yield rule in the set |
| Phone photos, skew, glare, thermal scans | ~20% of assumed real intake |
| Digital PDFs with a text layer | ~45% of assumed real intake; extracts near-perfectly and must be routed away from OCR |
| W-2c corrections, 4-up copy sheets | Distinct document handling |

Two mitigations. The Kaggle source includes noise-augmented images alongside clean
ones — pull those for the OCR-degradation arm. And apply Augraphy
(print/scan/photocopy artifacts) plus perspective, glare and JPEG noise to generate
a controlled degradation ladder, so accuracy can be reported *as a function of
image quality* rather than as a single number.

### Supporting open datasets

- **RVL-CDIP** — 400k document images, 16 classes. For the document-type
  classifier that decides "is this even a W-2" at intake.
- **FUNSD / XFUND** — form understanding with key-value and layout annotations.
  Layout pretraining for LayoutLM-family backbones.
- **`hsarfraz/donut-irs-tax-docs-classifier`** — Donut fine-tuned on 3,000+ IRS tax
  documents including W-2, built on `naver-clova-ix/donut-base-finetuned-rvlcdip`.
  A ready baseline for the intake router.

None of FUNSD, XFUND or RVL-CDIP contain W-2s. Citing them as if they did would be
misleading; they earn their place as pretraining and classification data only.

### Held to regardless

**Real hand-labeled documents remain the gold eval.** The public set is synthetic
from a single generator, so strong scores on it prove the model learned that
generator. Even 50 carefully labeled real documents are worth more as a final
check.

**Sampling bias:** if ground truth comes only from documents humans reviewed, and
humans review low-confidence documents, the eval set is biased. Force-sample a
random slice for full labeling.

---

## The A/B

Four arms, all scored on the same corpus:

| Arm | What it tests |
|---|---|
| **Template OCR** (Tesseract/PaddleOCR + coordinate mapping) | Cheap deterministic baseline; should dominate on clean digital PDFs |
| **Fine-tuned Donut** on the 1.8k train split | The dataset ships in exactly this format — the natural "we trained something" arm |
| **Hosted prebuilt** (Azure `prebuilt-tax.us.W2`) | Zero-training baseline; the honest question is whether training beats buying |
| **VLM structured JSON** (Claude / GPT-4-class, prompted to the schema) | Strongest zero-shot, expected to be badly overconfident |

**Shadow, paired, not split traffic.** Documents are asynchronous and cheap to
re-score, so every arm scores every document. Paired differences give far tighter
intervals on the same sample.

Reported per arm: field accuracy, critical-field accuracy, materiality-weighted
escape rate, reviewer-minutes per document, cost and p95 latency per document,
risk-coverage AUC, and an error taxonomy.

**Risk-coverage AUC is the metric most likely to change the decision.** It measures
whether confidence scores carry information at all. Expect the realistic signature:
the VLM scores higher raw accuracy but is badly overconfident, so no threshold
lets you trust its confidence to prioritize anything. Under mandatory sign-off,
confidence is what rations reviewer attention — selecting on field accuracy alone
picks the wrong arm. The harness should print that inversion explicitly.

**The reviewer model encodes automation bias.** A reviewer facing a prefilled form
with a green checkmark approves without reading; one whose attention is pointed at
a flagged box catches far more. Those two catch probabilities are the least
defensible numbers in the project and they drive the headline metric — which is why
`seeded_error_study.py` exists: inject known-bad values into a random share of the
queue and measure the real catch rate, split by whether the rule engine flagged the
field. Ship that instrumentation on day one so the assumption is falsifiable rather
than load-bearing and invisible.

Escape rate is weighted by dollar materiality throughout. Unweighted, a two-cent
rounding difference counts the same as a $4,000 error.

---

## Other assumptions

**A1 — Direction of flow.** A tax-prep/CPA firm whose clients *send W-2s in* for
1040 preparation. The alternative reading — a firm that *issues* W-2s for client
employees — is a generation-and-SSA-filing system (EFW2, BSO, W-3 transmittal) with
no extraction at all. The intake → extraction → follow-up shape only makes sense
for inbound documents. *If wrong:* the rule engine survives as a pre-filing gate,
which is why it has no dependency on the extractors.

**A2 — Intake mix.** Roughly 10–40k W-2s per season in a compressed
January–April window: ~45% digital payroll-provider PDFs, ~30% scans, ~20% phone
photos, ~5% unusable. **The load-bearing decision is to route on whether a text
layer exists before choosing a model.** The digital-PDF share extracts at near-100%
for near-zero cost; OCRing it anyway is the most common way these systems waste
money and introduce errors that were never in the document.

**A3 — Completeness is the hard problem, and it is not the W-9 problem.** With W-9s
the denominator is known: AP has a vendor list, so you know who owes you a form.
With W-2s there is no denominator — if a client sends two, nothing tells you
whether there was a third job. Three sources, descending reliability:

1. **IRS Wage & Income transcript** (signed Form 8821 or 2848, via e-Services/TDS).
   True ground truth. **Caveat:** not reliably complete until well after the filing
   deadline, so it is a post-filing reconciliation and amended-return trigger, not
   an in-season check. Assuming otherwise would be the biggest available design
   error.
2. **Prior-year rollforward.** Chase any EIN present last year and missing this
   year. Cheap, available in January, correct most of the time. The in-season
   workhorse.
3. **Secondary signals.** A state return implying wages we don't have; an organizer
   answer; a Box 12 code D implying a plan the client never mentioned.

Assumes ~85% returning clients. New clients have no denominator at all, and the
honest answer is an explicit attestation, not an inference.

**A4 — Security is non-negotiable.** Every W-2 is an SSN plus a full wage record.
GLBA Safeguards Rule, IRS Pub 4557, and a written information security plan for any
PTIN holder. Consequences: the follow-up agent **always** directs to a secure upload
link and **never** invites an email attachment; any third-party model API touching
document images needs zero-retention terms in writing first; field-level provenance
exists so that after a bad model version ships you can answer which documents were
affected.

---

## Architecture

```
w2/
  constants.py   per-tax-year IRS tables: wage base, 402(g), catch-ups,
                 valid Box 12 codes, no-income-tax states
  schema.py      canonical record; every value carries confidence + model version + bbox
  datasets.py    HF loader, gt_parse → canonical schema adapter, degradation ladder
  generate.py    payroll simulator for synthetic W-2s
  rules.py       declarative validation rules with severity and tolerance
  backends.py    Backend protocol + the four extraction arms
  evaluate.py    paired harness, reviewer model, risk-coverage curves
  chase.py       document state machine, follow-up cadence, completeness reconciliation
scripts/
  check_dataset_consistency.py   Experiment 0
  run_eval.py                    arm comparison table
  seeded_error_study.py          measures what the rule engine is worth
tests/
```

### Non-negotiable conventions

- **Money is integer cents.** Floats in a tax system are a correctness bug waiting
  to happen. The HF ground truth is strings with dollar signs and commas — parse to
  cents at the adapter boundary, once.
- **Per-year constant tables, never inline literals.** Every threshold on a W-2
  changes annually. An unknown tax year must raise loudly, not silently pass.
- **Repeated rows key on their natural key, never position.** Box 12 rows key on the
  code, state rows on the state code. Positional keying makes one dropped row look
  like an error in every row after it — it inflates error rates and hides the real
  failure mode.
- **Every value is a `Field`, not a scalar** — value, confidence, source model
  version, pixel bbox. Without the bbox you can't build a review UI a human can
  trust. Without the version you can't do incident response.

### Validation rules

Arithmetic: Box 4 from Box 3 + Box 7; Box 6 from Box 5 including additional
Medicare; Box 3 + Box 7 within the annual wage base; Box 5 − Box 1 explained by
Box 12 deferral codes (highest-yield rule — catches digit errors in either box plus
dropped and hallucinated Box 12 rows at once); no negative amounts; Box 2 not
exceeding Box 1.

Identifiers: SSN structurally issuable (no 000/666/9xx area, no 00 group, no 0000
serial; 9xx is an ITIN, never valid on a W-2); EIN well-formed.

Box 12/13: codes valid for the tax year; deferrals within 402(g) plus the largest
applicable catch-up; no duplicated codes; Box 13 consistent with Box 12.

State: no Box 17 withholding in no-income-tax states; Box 16 total in a loose band
around Box 1 (loose and warning-only on purpose — PA taxes 401(k) deferrals, so
state and federal wage bases legitimately differ); Box 17 plausible against Box 16.

Each rule carries an ID, severity, tolerance, a human-readable message, and **the
specific field names involved**, so the review UI can highlight exact boxes.

### Follow-up loop

`EXPECTED → REQUESTED → RECEIVED → EXTRACTED → FLAGGED → READY_FOR_REVIEW →
SIGNED_OFF`, plus `SUPERSEDED` (W-2c or better scan) and `ABANDONED` (client
confirms the job didn't exist). Illegal transitions raise.

Cadence escalates over four touches across ~4 weeks rather than nagging weekly;
the final touch requires human approval, then stop and escalate to the relationship
owner. Dedupe on `(SSN, EIN, tax year)` — Copy B, C and 2 are the same document and
clients re-upload clearer photos; keep the highest-confidence version. Review queue
orders by **expected dollar impact of an error**, not upload time.

---

## Known gaps, stated up front

- **Box 2 (federal withholding) is invisible to arithmetic.** No cross-field
  constraint exists on a W-2 — it depends on W-4 elections — so it falls back
  entirely on unaided human attention. That is a wrong refund. Needs a plausibility
  band from an effective-rate model, or a second independent extraction with a
  disagreement check.
- **SSN structural validation catches almost nothing.** Change one digit of a valid
  SSN and you usually get another valid SSN. The real control is name/SSN matching
  against the prior-year return, the client record, or IRS TIN matching — not a
  regex. Shipping the regex and *believing SSNs are covered* is the dangerous
  outcome.
- **W-2c corrections** — out of scope for v1. `SUPERSEDED` exists; the parser
  doesn't.
- **Multi-form page segmentation** — Copy B/C/2 print 4-up. Partially handled at
  dedupe, which is not a full answer.
- **W-3 / 941 cross-document reconciliation** — belongs to the payroll-filer
  reading of A1. Highest-value addition if the firm also does payroll.

---

## Build order

1. `constants.py`, `schema.py` — foundations everything types against.
2. `datasets.py` + `check_dataset_consistency.py` — load the HF set, adapt
   `gt_parse` to the canonical schema, run Experiment 0. **Do this before writing
   any rule**, because the answer determines how much the generator matters.
3. `generate.py` + `rules.py` together, until zero false positives on clean
   generated documents.
4. `evaluate.py` + `seeded_error_study.py` — the harness.
5. `chase.py` — state machine and completeness reconciliation.
6. The four extraction arms behind the `Backend` protocol; re-run the harness
   unchanged.
7. A control for Box 2, then name/SSN matching — the two largest uncovered
   exposures.

---

## Production architecture (design only, not v1)

Immutable artifacts in object storage with content-hash IDs. Durable orchestration
with human-in-the-loop wait states — Temporal handles multi-week chase timers well.
Field-level provenance so every value traces to a model version and a pixel region.
Per-year rule configuration versioned alongside the constant tables, so a January
threshold change is a config commit rather than a deploy.

---

## Open questions

1. What is the **real intake mix**? If it's 90% phone photos, preprocessing becomes
   the main investment and the cost model above is wrong.
2. What fraction of clients are **returning**? Rollforward is the entire in-season
   completeness strategy and only works for them.
3. Does the firm hold **8821/2848 authorizations** at scale? Determines whether
   transcript reconciliation is available at all.
4. What is the **measured** unflagged catch rate in review? If it's low, the finding
   is that the review queue isn't the safety net the firm believes it is — and the
   fix is the UI, not the model.
