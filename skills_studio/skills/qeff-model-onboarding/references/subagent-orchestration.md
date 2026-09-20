# Subagent Onboarding Orchestration

Use this protocol when a new-model onboarding task is delegated to multiple
agents. Its purpose is to prevent a successful export, compile, or token sample
from masking a PyTorch, cache, or architecture-integration regression.

## Roles

### Coordinator — GPT-5.5

Owns the state machine, evidence ledger, and rerouting. The coordinator does
not implement model code and does not approve its own design review.

- Opens the task by requesting the model id/config, `HF_HUB_CACHE` before any
  download, the virtualenv before Python commands, and QAIC availability only
  when the QAIC stage is reached.
- Asks the architect for a plan for the active stage, gives that plan to the
  executor, then gives the resulting diff and evidence to the verifier.
- Advances a representative only when the current stage passes. It never
  converts a missing dependency, unavailable hardware, skipped test, or compile
  success into a parity pass.
- Routes a verifier failure back to the architect when the design, feature
  grouping, ownership, or validation strategy is wrong; otherwise it returns a
  focused fix to the executor.
- Stops blind retrying after two failed attempts with the same stated root
  cause. Report the evidence and request human direction instead.

### Architect — GPT-5.6 Sol

Plans one active stage at a time and makes no code changes. It must read the
model-family map and the closest local wrappers before proposing work.

For each handoff, provide:

- upstream architecture inventory and closest local family/files;
- a semantic-layer coverage matrix and selected representative layer indexes;
- the exact behavior being changed, compatibility surfaces preserved, and files
  expected to change;
- test inputs that activate the path, including prefill and decode/cache cases
  where relevant;
- the stage-specific command(s), expected comparison outputs, and exit
  criterion;
- known unsupported paths and the evidence required to call them blocked.

The architect must not prescribe a broad copy of an upstream model file or
generic/base-class changes without proving that every affected model needs them.

### Executor — GPT-5.6 Terra

Implements only the approved active-stage plan. It owns code and tests, not
stage approval.

- Reuse the named local family and keep model-specific behavior in the model
  wrapper unless the plan establishes shared ownership.
- Add focused coverage to an existing test location, preferring
  `tests/test_model_quickcheck.py` where it carries the regression.
- Run the planned narrow validation, format/lint affected code, and return the
  diff, commands, pass/fail output, observed numerical differences, and any
  generated artifact paths.
- Do not start ORT or QAIC work while the corresponding QEff PyTorch gate is
  failing. Do not add speculative fallback paths to make a later stage run.

### Verifier — GPT-5.5

Independently examines the diff and evidence. It does not rewrite the
implementation. Every failure must cite a concrete path, line, comparison,
missing activation, or violated local precedent; it must not emit preference
only feedback.

Check the changed modeling code against the closest local implementations for:

- attention projection layout, head/KV grouping, masking, cache update/indexing,
  prefill versus decode behavior, sliding-window/chunking, and layer windows;
- MLP activation/gating, MoE routing/top-k normalization, expert weight
  transforms, and prefill/decode MoE variants;
- RMSNorm or other norm epsilon/weight semantics, RoPE scaling/position ids,
  rotary dimensions, and multimodal embeddings where present;
- transform registration, Auto/export/runtime glue, graph input/output names,
  cache layout compatibility, and model-specific logic leaking into shared code;
- whether the selected tests actually activate every coverage-matrix row.

Return one of `PASS`, `FIX_EXECUTOR`, `REPLAN_ARCHITECT`, or `BLOCKED`, with a
short evidence table. A verifier cannot approve ORT or QAIC parity solely from
export/compile success.

## Coverage Matrix

Before implementation, inventory each distinct execution-semantic class:

| Surface | Representative selection |
| --- | --- |
| Attention | One layer per attention implementation, plus every transition between layer policies. |
| MLP / MoE | One layer per dense MLP and each routed-expert implementation; include router behavior and every distinct expert layout. |
| Normalization | One layer per norm type or placement convention. |
| Position encoding | One layer per RoPE/position implementation or scaling regime. |
| Cache | A prefill case and a decode case for each cache layout or update policy. |
| Other | One representative for each multimodal tower, hybrid/SSM block, chunking, sliding-window, or compression path. |

Also include the first and final decoder layers when they differ in observable
behavior, or when that is necessary to cover final normalization, logits, or a
layer-window boundary. A homogeneous decoder may use one representative layer;
a mixed architecture must use enough layers to exercise every row. The smallest
valid test model is therefore feature-complete, not simply one layer deep.

For attention/cache coverage, use a prefill sequence length of at least two and
at least one follow-up decode token. For routed MoE, choose inputs/configuration
that exercise routing rather than a degenerate single-expert or no-op route.

## Stage Machine

The coordinator maintains one row per coverage-matrix representative:

| Stage | Entry gate | Required evidence | Exit |
| --- | --- | --- | --- |
| `PYTORCH` | implementation and representative test exist | HF PyTorch and transformed QEff PyTorch compared on identical eval-mode inputs, weights, dtype, prefill, and decode/cache states | numerical comparison and cache/output contract pass |
| `ORT` | `PYTORCH` passed | exported ONNXRuntime output compared with the same HF reference for the active representative; retain-state outputs and names are checked | numerical comparison and retained-state contract pass |
| `QAIC` | `ORT` passed and a QAIC environment/QPC is available | compiled QAIC run compared with the same HF reference for active representative, including cache progression when applicable | numerical comparison and runtime cache contract pass |

Use existing local comparison helpers and tolerances; do not invent a relaxed
threshold to clear a new model. Record dtype, shapes, max absolute difference,
max relative difference when available, and the exact command for every pass.

If ORT is unsupported for the model category, or QAIC hardware/compiler access
is unavailable, mark that row `BLOCKED` with the exact reason and attempted
command. The final report may say PyTorch support passed and ORT/QAIC remains
unverified; it must not say the full model is supported on QAIC.

## Coordinator Loop

1. Classify the upstream model and ask the architect for the coverage matrix and
   `PYTORCH` plan.
2. Send the approved plan to the executor. The executor implements and returns
   code/test evidence.
3. Send the plan, diff, and evidence to the verifier.
4. On `PASS`, move the same representative to the next permitted stage. On
   `FIX_EXECUTOR`, give the verifier's concrete finding to the executor. On
   `REPLAN_ARCHITECT`, request a new plan for the same stage. On `BLOCKED`,
   record the capability gap and continue only with independent, still-valid
   work.
5. After all representatives have reached their terminal state, request a final
   verifier pass over the aggregate diff and coverage ledger. Run the relevant
   quickcheck scope and widen it when shared cache, Auto, export, or runtime
   code changed.

## Handoff Format

Every agent response must identify:

```text
model:
coverage_rows:
active_row:
stage:
decision: PASS | FIX_EXECUTOR | REPLAN_ARCHITECT | BLOCKED
evidence:
  commands:
  comparison:
  changed_files:
  artifacts:
next_owner:
```

This is an evidence record, not a status narrative. Do not claim a stage passed
without the command and comparison that establish it.
