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

## F3 — (reserved) Rule engine false-positive rate on clean synthetic data

Acceptance target: zero findings across 3,000 generated documents.

## F4 — (reserved) Extraction arm comparison

## F5 — (reserved) Seeded-error study: measured value of the rule engine
