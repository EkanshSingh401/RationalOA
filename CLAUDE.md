# CLAUDE.md

## Project
W-2 automation pipeline. See README.md for the full spec and assumptions.

## Non-negotiable conventions
- Money is ALWAYS integer cents. Never float. Parse to cents at the
  adapter boundary, once.
- Tax constants live in w2/constants.py keyed by year. Never inline a
  threshold. Unknown tax year must raise, never silently pass.
- Repeated rows key on their natural key, never position:
  box12[D]_amount, state[GA]_box17_tax. NOT box12_0_amount.
- Every extracted value is a Field (value, confidence, source, bbox).
  Never a bare scalar.
- Standard library + pytest only unless I approve a dependency.

## Working style
- Do the step I asked for, then stop and report. Don't run ahead.
- Write the test before or alongside the code, not after.
- If you find something surprising in the data, tell me before working
  around it.