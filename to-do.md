# Deferred refactoring — instructions for future work

Status: deferred, not automatically authorized. Phase 1 is the current scope.
Before continuing, read README.md, baseline-v00.md and to-solve.md, inspect the
current implementation, and confirm scope with the researcher. Do not silently
select scientific methods. This project and its analyses are under development.

## Non-negotiable working style

- Ordinary `.py` files, executed line by line with Ctrl+Enter in the researcher's
  interactive window; variables and plots remain inspectable alongside the editor.
- No notebook migration and no `# %%` cell markers.
- Plain headings, whitespace, descriptive names and explicit scientific steps.
- Use V00 as the development recording. Checks must remain understandable without
  pytest, unittest, fixtures, CI knowledge or an additional testing dependency.
- Preserve reminders and unresolved choices in to-solve.md.
- Keep reusable functions, but do not hide the workflow behind a large function
  or introduce a pipeline framework, class hierarchy or configuration framework.

## Phase 2 — Make the interactive scripts consistent

Prerequisite: review Phase 1 findings and the saved-ROI identity mismatch.

- [ ] Format scripts with plain headings and small explanatory blocks, not cells.
- [ ] Keep subject/mode/run choices visible near the top.
- [ ] Separate inspection, decision editing, reload, application and explicit save.
- [ ] Provide single-subject feature exploration before group collection.
- [ ] Define safe rerun points and explain which objects are modified in place.
- [ ] Preserve scientific order; any method change needs a separate decision.

Acceptance: researcher can select lines, inspect intermediate variables and plots,
and restart/run sequentially without entering a large function. No duplicate
scientific pipeline or notebook is introduced.

## Phase 3 — Trial identity and output provenance

Prerequisite: agree block boundaries and output migration with researcher.

- [ ] Establish an original-event table; distinguish original identity from position.
- [ ] Carry identities and inclusion information through EEG, EMG, ROI and exports.
- [ ] Introduce mode/run-aware paths without silently invalidating existing files.
- [ ] Save effective settings, decision snapshot, inputs and software/code versions.
- [ ] Make overwrite choices explicit, including figures and database export.

Acceptance: trace derived results to input events and processing decisions. Do not
infer blocks by dividing every 200 events or change inclusion rules.

## Phase 4 — Simplify reusable code

Prerequisite: preserve numerical baselines and resolve affected method questions.

- [ ] Consolidate window/peak implementations after agreement on boundaries/polarity.
- [ ] Separate event interpretation, decision records and MATLAB export duties.
- [ ] Centralize repeated path construction.
- [ ] Return plot figures; make display/save/close behavior explicit.
- [ ] Configure plotting backend before plotting imports; avoid remote=notebook assumptions.
- [ ] Reduce private-library API use where possible, verifying numerical equivalence.
- [ ] Keep the configuration dataclass unless a concrete problem warrants splitting it.
- [ ] Consider MNE-compatible epoch filenames with an explicit compatibility plan.

Acceptance: one implementation per agreed calculation; no gratuitous class rewrite,
tiny-module proliferation or hidden scientific operations.

## Phase 5 — Optional productivity improvements

Prerequisite: stable single-subject interactive workflow; researcher approves scope.

- [ ] Small batch runner reusing existing operations, not a second pipeline.
- [ ] Save useful review checkpoints to avoid expensive recomputation.
- [ ] Consider compact processing summaries/reports.
- [ ] Discuss automation only if useful; CI and testing frameworks are not requirements.

Acceptance: productivity improves without making local inspection or debugging harder.