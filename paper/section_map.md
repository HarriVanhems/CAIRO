# Section Map for Paper Agents

Canonical source: `tex_CAIRO/main.tex`

This file is for assignment only. Agents may read `main.tex` directly, but parallel edits should happen in `paper/staging/` files first and be merged back manually.

## Recommended Review Blocks

- `intro_background`
  lines `121-184`
  covers Introduction and Background and Related Work

- `framework_calibration`
  lines `185-399`
  covers the decoupled framework and the two main theoretical stages

- `algorithms`
  lines `400-552`
  covers estimation details and alternative monotone recalibration maps

- `survival_main`
  lines `553-720`
  covers the main survival extension section and its three calibration schemes

- `experiments`
  lines `721-974`
  covers synthetic setup, synthetic results, and real-data analysis

- `discussion`
  lines `975-996`
  covers the discussion/conclusion section

- `proofs_main`
  lines `997-1665`
  covers the main technical appendix and theorem proofs

- `proofs_cox`
  lines `1666-2131`
  covers the Cox extension proofs and IPCW-Gini derivations

- `survival_appendix_tail`
  lines `2132-2240`
  covers the final survival appendix block

## Parallel Assignment Rules

- Review-only agents can read any block directly from `tex_CAIRO/main.tex`.
- Rewrite agents should own exactly one block at a time.
- Do not assign two rewrite agents to overlapping line ranges.
- Keep proofs separate from narrative sections; they need a different review standard.

## Current Structural Note

- `Extension to Survival Analysis: Calibrating Right-Censored Outcomes` appears twice in `main.tex`, once in the main body and once again near the end. Keep those assignments separate so agents do not conflate them.
