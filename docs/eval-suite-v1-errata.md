# Eval suite v1 — errata

This file is the v1 errata log. Per `docs/eval-suite-v1.md`'s
versioning policy:

> Bugs in v1 don't get fixed in v1. If a Tier 2 task turns out to be
> broken (e.g., the reference solution is wrong), we note it here and
> live with the noise. Editing v1 mid-experiment is forbidden.

That rule is load-bearing for the experiment integrity: once we let
the eval suite respond to model behavior, the model's score stops
measuring anything external. So bugs go here, not into the suite.

## When to file an erratum

File one when:
- A Tier 2 task's reference solution is later discovered to be wrong.
- A held-out problem's schema admits the "wrong" answer (e.g.,
  empty {} unexpectedly passes).
- A rubric runner has a deterministic bug (e.g., counts an import
  twice, misses a valid call form).
- The aggregator's weighting math drifts (catch this in CI; the
  errata records the incident).

**Do not** file an erratum for:
- "The model scored badly on Task X." That's not a bug; it's a result.
- "The reference solution is a *style* I disagree with." Style isn't a
  correctness criterion.
- A failed jsonschema validation against a malformed model response.
  That's the model failing, not the suite.

## Erratum schema

```markdown
## Erratum N

- **Severity**: low | medium | high | critical
- **Date discovered**: YYYY-MM-DD
- **Affected component**: <module / task_id / rubric / aggregator>
- **Discovery**: how it was found
- **Description**: what the bug is, in one paragraph
- **Reproduction**: minimal steps if applicable
- **Decision**: live-with-noise | ship-v2 | hotfix-allowed-because-X
- **Generation impact**: list affected G_N runs, or "none yet"
```

**Severity rubric:**

| Severity | Meaning                                                          |
|----------|------------------------------------------------------------------|
| low      | One task slightly mis-scores; sub-1-point impact on Tier 2 total |
| medium   | One task off by its full point value; up to ~3 points on T2     |
| high     | Multiple tasks affected; rubric runner systematically off       |
| critical | Score comparability between G_N and G_(N+1) breaks              |

**Decision rubric:**

- `live-with-noise`: default. The bug stays, the score absorbs the
  noise. Document everything so future-us can re-analyze.
- `ship-v2`: severity is high or critical. Open a v2 issue, freeze
  v1, restart with a fresh G0 lineage on v2.
- `hotfix-allowed-because-X`: extremely rare. Only when the bug
  makes the suite literally non-runnable (e.g., a syntax error in a
  reference solution that crashes the runner before scoring starts).
  Document the exact harm a strict reading would have caused.

## Erratum log

*(none yet — v1 is freshly locked as of 2026-05-17.)*

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/2347/
