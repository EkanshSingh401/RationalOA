# Findings

Running log of things measured rather than assumed. Each entry: what was tested,
what came back, what it changes. The final README pulls from this.

---

## F1 — The public W-2 dataset has systematically wrong FICA arithmetic

**Dataset:** `singhsays/fake-w2-us-tax-form-dataset` (2,000 records; 1.8k/100/100
split; ~1.2k downloads/month at time of writing).

### How it was found

Experiment 0 tested whether the dataset's amounts satisfy real payroll arithmetic
before any rules were written. The naive check looked conclusive and was wrong:

| Check | Within $1 | Mean residual | Stdev |
|---|---|---|---|
| A: `Box4 − 0.062 × (Box3 + Box7)` | 0/2000 | −$6,944.82 | $3,188.57 |
| B: `Box5 − Box1 − Σ Box12 deferrals` | 0/2000 | −$4,674.20 | $27,587.67 |

Formula B straddles zero with a huge spread — consistent with unrelated random
values. Formula A did not: **every** record was short, never over, with a tight
spread relative to the mean. One-sidedness that clean is structure, not noise.

The cause was a defect in a *different* field. Box 7 ≡ Box 3 and Box 8 ≡ Box 5 in
all 2,000 records (the generator copies rather than generating tips), so
`Box3 + Box7` double-counts wages and the residual is forced negative regardless of
whether Box 4 is correct.

**Lesson worth keeping:** a contaminated denominator produced a confident,
plausible, and wrong conclusion. The tell was the sign distribution, not the pass
rate.

### What is actually true

Re-tested on ratios, including `Box6/Box5` which no duplicated field touches:

| Ratio | Observed | Stdev | Correct value |
|---|---|---|---|
| `Box4 / Box3` | 0.076500 | **0.000000** | 0.062 |
| `Box6 / Box5` | 0.029000 | **0.000000** | 0.0145 |

Zero variance across 2,000 records. Box 4 = `round(0.0765 × Box3)` and
Box 6 = `round(0.029 × Box5)` hold to the cent in 1,998/2,000 records (max error
1¢, i.e. rounding).

The coefficients are not arbitrary:

- **0.0765 = 0.062 + 0.0145** — the *combined* employee FICA rate, applied where
  only the Social Security component belongs.
- **0.029 = 2 × 0.0145** — the *employer + employee* Medicare rate, applied where
  only the employee share belongs.

Both are the same class of error: reaching for the wrong rate from the right
family. This is a deterministic, reproducible generator bug, not random data.

### The more serious defect: no wage base cap

| Threshold | Records above |
|---|---|
| Box 3 > $168,600 (2024 SS wage base) | 753 / 2000 (37.7%) |
| Box 3 > $176,100 (2025 SS wage base) | 695 / 2000 (34.8%) |

Box 4 shows no cap effect at either threshold — it stays a flat function of Box 3.
The generator never modeled the cap.

**This matters more than the rate errors.** A wrong rate is a constant factor and
is recoverable by division. A missing cap destroys information: real Box 3 is
`min(wages, base)`, so a capped value cannot be inverted. Those ~695 records are
not merely wrong, they are structurally impossible — no real W-2 has Box 3 above
the wage base.

It also explains the Box1/Box3 correlation of 0.917 rather than something tighter.
With a real cap, high earners pile up at exactly the base and the relationship
kinks; here it stays smooth.

### Third defect: Box 3 exceeds Box 1

| | min | max | mean | stdev |
|---|---|---|---|---|
| Box 1 (wages) | $40,109.45 | $249,948.53 | $146,197.21 | $61,558.52 |
| Box 3 (SS wages) | $28,777.55 | $313,153.32 | $146,206.69 | $67,127.88 |
| Box 5 (Medicare wages) | $30,862.86 | $322,812.86 | $145,134.76 | $66,727.53 |

Correlations: Box1–Box3 0.917, Box1–Box5 0.912, Box3–Box5 0.838. Correlated, so
drawn from a shared base salary with per-box noise — not independent draws.

Box 3 legitimately exceeds Box 1 when pre-tax deferrals exist, but it is still
capped. Box 3 simultaneously above Box 1 *and* above the wage base cannot occur.
Formula B's ±$83k spread comes from records where Box 1 exceeds Box 5, which is
also impossible (Box 5 = Box 1 + deferrals).

### Consequences

1. **The dataset is usable for layout and OCR, not for tax arithmetic.** Images and
   field positions are fine. The numbers are not a valid source of payroll truth.
2. **Anyone fine-tuning on this corpus is teaching a model a wrong tax world** — that
   Box 4 is 7.65% of Box 3 and that Box 3 has no ceiling. The model will reproduce
   it faithfully. This is the concrete argument for keeping the validation layer
   downstream of extraction regardless of extractor quality.
3. **The rule engine is validated against real data before being written.** Expected
   fire rates on the public corpus: `SS_TAX_MISMATCH` ~100%,
   `MEDICARE_TAX_MISMATCH` ~100%, `SS_WAGE_BASE_EXCEEDED` ~35%. Confirmed by
   `scripts/score_public_dataset.py`.
4. **Added rule:** `BOX3_EXCEEDS_BOX1_UNEXPLAINED`, prompted by this data.

---

## F2 — The dataset has no tax year field

All 2,000 records share an identical 45-key `gt_parse` set, and none of them
carries the tax year. Missing values are the literal string `"None"`, never JSON
null and never an absent key.

The adapter takes `tax_year` as a **required** caller-supplied argument and stamps
it `Field(value, confidence=0.0, source="assumed")` rather than pretending it was
extracted.

This is a real limitation of the corpus. On a genuine W-2 the year is printed on
the form, and in production it is an extraction target that selects which constant
table to load — so a dataset omitting it cannot exercise that path at all.

---

## F3 — Rule engine: false-positive rate on clean data, true-positive rate on the public corpus

**Acceptance target:** zero findings across a few thousand generated documents
(README, "Secondary: the payroll-consistent generator").

### Result: 48/48 tests passing, zero findings on generated data

Zero findings across **23,000 generated documents spanning 21 seeds** — the
pinned 3,000-document acceptance test (seed 42) plus a 20×1,000 stress sweep
(seeds 0–19), re-verified after the two fixes below.

### Two bugs found and fixed, from two different sources

**1. Generator: degenerate Box 1, caught directly by the acceptance test.**
`_generate_box12` sized the 402(g) deferral budget off the year's dollar limit
alone, with no regard for how small `gross` was. A low-earner record could be
assigned a near-max deferral, and the negative-Box-1 backstop clamped it to a
technically-non-negative but degenerate ~$0.01 Box 1 — which then tripped
`STATE_WAGES_OUT_OF_BAND` when the record was validated. Fixed by capping the
deferral budget at half of available wages, not just the 402(g) limit, so the
backstop clamp is now essentially never exercised instead of being load-bearing.

**2. Schema: `flat_fields()` Box 12 collision, an evaluation-integrity bug,
not caught by the acceptance test.** The generator only ever emits distinct
Box 12 codes (`rng.sample`, no replacement), so this could never surface
there — it was flagged during rule design (writing `BOX12_DUPLICATE_CODE`)
and confirmed materially real by scoring the public corpus, where 611/2,000
records duplicate a code (three `E` entries is common). `flat_fields()` kept
one dict entry per *code*, so a duplicated code silently dropped every entry
but the last — which matters beyond labeling, because `flat_fields()` is what
an eval harness diffs truth against prediction on: two real amounts would
never enter the denominator, and a score computed against a shrunk
denominator looks better than it is. Fixed by numbering entries within a
duplicated code (`box12[E#1]_amount`, `box12[E#2]_amount`, ...), ordered by
amount descending rather than source order, since Box 12 slot position
(12a–12d) carries no meaning and an extractor returning the same rows in a
different order must not be scored as an error. A code appearing once keeps
the plain `box12[<code>]_amount` form.

Also removed the second clause of `BOX3_EXCEEDS_BOX1_UNEXPLAINED`
("Box 3 above Box 1 and above the wage base cannot occur"): if Box 3 exceeds
the wage base then Box 3 + Box 7 also does (Box 7 >= 0), so
`SS_WAGE_BASE_EXCEEDED` already fires on every record that clause would have
caught. It only added a duplicate finding on the same underlying defect —
inflating flag counts and charging the reviewer real time for a box they were
already sent to. Measured effect on the public corpus: 924/2,000 (46.2%) to
901/2,000 (45.1%) — 23 records had been flagged by that clause alone.

### Public-dataset fire rates (`scripts/score_public_dataset.py`, post-fix)

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
| `SSN_MALFORMED` | CRITICAL | 0/2000 (0.0%) |
| `SSN_INVALID_AREA` | CRITICAL | 0/2000 (0.0%) |
| `SSN_INVALID_GROUP_SERIAL` | CRITICAL | 0/2000 (0.0%) |
| `EIN_MALFORMED` | CRITICAL | 0/2000 (0.0%) |
| `BOX12_INVALID_CODE` | ERROR | 0/2000 (0.0%) |
| `BOX12_OVER_402G` | WARN | 0/2000 (0.0%) |
| `BOX12_DUPLICATE_CODE` | WARN | 611/2000 (30.6%) |
| `BOX13_RETIREMENT_INCONSISTENT` | WARN | 1238/2000 (61.9%) |
| `NO_TAX_STATE_WITHHOLDING` | ERROR | 701/2000 (35.0%) |
| `STATE_WAGES_OUT_OF_BAND` | WARN | 0/2000 (0.0%) |
| `STATE_TAX_IMPLAUSIBLE` | ERROR | 0/2000 (0.0%) |

`SS_WAGE_BASE_EXCEEDED` at 77.5% is not the ~35% Box 3 alone would suggest
(and did suggest, in F1's threshold table). The rule checks Box 3 + Box 7 per
spec, and Box 7 is set identically equal to Box 3 in every record (F1's
tips-duplication artifact), so the checked total is effectively 2x Box 3 —
roughly doubling the fire rate. A rule-input artifact carried over from F1,
not a new rule bug.

`BOX1_EXCEEDS_BOX5` at 52.2% is a near-coin-flip. That's independent
confirmation — from the rule-firing side rather than the correlation-and-sign
analysis F1 used — of the same conclusion: Box 1 and Box 5 are not derived
from each other in this dataset, they're drawn independently, so which one
happens to be larger is close to a fair coin.

Every identifier/format rule (`SSN_MALFORMED`, `SSN_INVALID_AREA`,
`SSN_INVALID_GROUP_SERIAL`, `EIN_MALFORMED`, `BOX12_INVALID_CODE`,
`BOX12_OVER_402G`, `STATE_WAGES_OUT_OF_BAND`, `STATE_TAX_IMPLAUSIBLE`) sits at
0%, because Faker produces well-formed SSNs and EINs and stays within
realistic Box 12 magnitudes and state-wage ratios. Net effect: this public
corpus exercises and validates the arithmetic layer of the rule set and
essentially nothing else — the identifier and structural rules are unproven
against real-world malformed input by this corpus alone.

## F4 — (reserved) Extraction arm comparison

## F5 — (reserved) Seeded-error study: measured value of the rule engine
