# Slop catalogue

Reference for AI-slop and quality patterns in QEfficient PRs. Each entry pairs a POSIX-safe grep with severity and fix recommendation.

## Table of contents

- Surface artifacts: [committed debug files](#committed-debug-artifacts), [copy-named files](#files-literally-named-copy), [shebang on library module](#library-module-with-shebang), [missing license header](#missing-license-header-on-new-py), [missing trailing newline](#missing-trailing-newline-on-new-py)
- Debug code: [pdb/breakpoint](#debug-calls-in-production-code), [print in library](#print-in-library-code), [unicode emoji in code](#unicode-emoji--box-drawing-in-code)
- Imports: [silent fallbacks](#silent-fallback-on-import), [bare/wildcard except](#bare-except--exceptpass), [hallucinated transformers symbols](#hallucinated-transformers-symbol), [wildcard import in library](#wildcard-import-in-library), [imported but never used](#imported-but-never-used)
- Feature toggling: [env-var feature toggle](#feature-toggled-by-environment-variable), [class-level mutable state](#class-level-mutable-state-shared-across-instances), [comment-to-toggle in examples](#toggle-by-commenting)
- Architecture: [change in wrong layer](#change-landed-in-the-wrong-layer), [unmapped QEff class](#new-qeff-class-not-registered-in-pytorch_transformspy), [hash plumbing missed](#hash-invariant-break), [PrefillOnly/Revert asymmetry](#prefillonlyrevert-asymmetry), [generic↔specific coupling](#cross-class-state-coupling), [misplaced helpers in utils](#misplaced-helpers-in-qefficientutils), [duplicate function definitions](#duplicate-function-definitions), [destructive self-mutation in forward](#destructive-self-mutation-in-forward), [hardcoded dtype](#hardcoded-compute-dtype), [hardcoded masked-attention fill](#hardcoded-masked-attention-fill-value)
- Comments / docstrings: [commented-out code](#commented-out-imports--code), [TODO for current PR work](#todofixme-for-things-that-should-be-done-now), [PR/review-meta comments](#comments-referencing-the-prreview), [restating-code comments](#comments-restating-the-code), [AI-style docstrings](#ai-style-multi-paragraph-docstrings-on-trivial-getters), [docstring/signature mismatch](#docstring-args-dont-match-signature)
- Examples: [duplicated example files](#duplicated-example-files), [magic numbers](#magic-numbers-in-models--examples), [private-name imports in examples](#examples-importing-private-names)
- Tests: [hand-built PKV "parity"](#hand-built-past_key_values-to-test-parity), [smoke-as-parity](#smoke-test-masquerading-as-parity), [parameter that defeats the test](#test-that-fixes-a-parameter-so-the-buggy-path-doesnt-run), [config shadowed by skip-list](#config-entry-shadowed-by-a-skip-list), [test that monkey-patches the SUT](#test-monkey-patches-the-sut), [TBD validate.md row](#validatemd-row-that-is-just-tbd--)
- Process: [WIP/Draft proposed for merge](#wip--draft-pr-proposed-for-merge)
- Hygiene: [trailing whitespace](#trailing-whitespace-lines), [imports out of isort order](#imports-out-of-isort-order), [lint suppression with feature](#lint-suppression-added-with-feature-code), [assert without message](#assert-without-message), [mass cosmetic+feature mix](#mass-cosmetic-changes-mixed-with-feature-work), [empty PR description](#empty-pr-description--vague-title), [many lint-fix commits](#many-lint-fix--fix-ci-commits)
- PEP 8 / structural: [oversized method](#oversized-or-multi-responsibility-method), [deep nesting / cognitive complexity](#deep-nesting--cognitive-complexity), [misleading or generic names](#misleading-or-generic-names), [mutable default arg](#mutable-default-arg)

## Severity grading rule

- **B (Blocker)** — would land a bug, contract violation, or shipped slop in main. Always block.
- **I (Issue)** — degrades the codebase but unlikely to ship a wrong result. Should be fixed before merge.
- **N (Nit)** — cosmetic/style. Mention without gating.

If you find ≥1 Blocker, the verdict is `request-changes`. If 0 Blockers and ≥4 Issues, the verdict is `comment` (or `request-changes` if those Issues collectively undermine confidence). Clean PRs (0/0/0 or only Nits) get `approve`.

## Catalogue

### Committed debug artifacts [B]

```bash
grep -nE "^diff --git a/(dbg\.log|.*\.pyc|.*\.log|.*__pycache__|.*\.swp|.*\.bak|core\.[0-9]+) " /tmp/pr<N>.diff
```

Empty-file additions don't have `+++ b/<path>` lines, so the grep keys on `diff --git a/`. Catches `dbg.log`, `*.pyc`, `__pycache__/`, `.swp`, `.bak`, `core.<pid>` — any of these in the file list reveal the author hand-staged past `.gitignore`. Fix: `git rm`, then add to `.gitignore` if not already.

### Files literally named "copy" [B]

```bash
grep -nE "^\+\+\+ b/.*( copy| copy 2|_copy|\(copy\)|[Cc]opy_of_)\.py" /tmp/pr<N>.diff
```

Matches `modeling_X copy.py`, `modeling_X (copy).py`, `modeling_X_copy.py`, `Copy_of_modeling_X.py`. The bare alternation `|copy)` would false-positive on `cache_utils.py`/`file_copy.py`/etc., so the regex requires explicit ` copy` / `_copy` / `(copy)` / `Copy_of_` boundaries. Fix: delete the copy file.

### Debug calls in production code [B]

```bash
grep -nE "^\+.*\b(p[d]b\.set_trace|break[p]oint\(\)|i[p]db\.set_trace|IPython\.embed)" /tmp/pr<N>.diff
```

Catches all debugger-entry calls: the `pdb`/`i`+`pdb` `.set_trace` family, bare `break`+`point()`, and `IPython.embed()`. Watch for env-gated forms (`if os.getenv("QEFF_DBG"): ...`) — those still ship a debugger import in library code; treat as Blocker even though they look "guarded". Fix: delete all.

### `print()` in library code [I]

```bash
grep -nE "^\+\s*print\(" /tmp/pr<N>.diff | grep -vE "(examples/|scripts/|tests/)"
```

Library code uses `QEfficient.utils.logging_utils.logger`. `print()` calls under `examples/`, `scripts/`, `tests/` are fine. Fix: switch to `logger.info` / `logger.debug` or delete.

### Unicode emoji / box-drawing in code [I]

```bash
grep -nP "^\+.*[\x{2500}-\x{27BF}\x{1F300}-\x{1FAFF}]" /tmp/pr<N>.diff
```

`print("✓ qaic_config…")`, `print("─" * 80)`, etc. Some terminals corrupt logs and grep/awk pipelines choke on these bytes. Strong AI-author tell. Fix: ASCII-only strings.

### Missing license header on new `.py` [B]

```bash
for f in $(git diff --name-only --diff-filter=A <base>..HEAD -- '*.py'); do
  grep -q "SPDX-License-Identifier: BSD-3-Clause" "$f" || echo "MISSING HEADER: $f"
done
```

The repo uses BSD-3-Clause (not Apache despite older contributor docs). Use a `grep -q` for the SPDX line — small dash-count drift in the canonical 7-line block exists in main and a byte-equal check would false-positive. Fix: prepend the canonical header (any committed file with a valid header is a fine source).

### Missing trailing newline on new `.py` [N]

```bash
grep -E "^\\\\ No newline at end of file" /tmp/pr<N>.diff
```

Pre-commit normally adds the trailing newline. Its absence here means the author hand-edited after pre-commit or bypassed it. Fix: add a trailing newline.

### Library module with shebang [I]

```bash
for f in $(git diff --name-only --diff-filter=A <base>..HEAD -- 'QEfficient/**/*.py'); do
  head -1 "$f" | grep -q "^#!" && echo "SHEBANG ON LIBRARY: $f"
done
```

`#!/usr/bin/env python3` on a library module that's imported, not executed, is wrong. Reserve shebangs for `examples/` and `scripts/`. Often co-occurs with missing license header. Fix: drop the shebang, prepend the BSD-3-Clause header.

### Silent fallback on import [I]

```bash
grep -nE "^\+\s*except\s+\(?(ImportError|ModuleNotFoundError|RuntimeError)" /tmp/pr<N>.diff
```

```python
try:
    from QEfficient.diffusers.pipelines.flux import QEffFluxPipeline
except (ImportError, RuntimeError):
    QEffFluxPipeline = None
```

Real install failures and runtime errors get hidden behind `None`; users hit `TypeError: 'NoneType' object is not callable` later instead of the actual error. The narrow `except ImportError` form for a genuinely optional dep is acceptable when paired with a clear error at first use. `except (ImportError, RuntimeError)` is almost always too broad. Fix: narrow the catch, log a warning, surface a useful error at the use site.

### Bare `except` / `except: pass` [I]

```bash
grep -nE "^\+\s*except\s*:" /tmp/pr<N>.diff
grep -nA1 "^\+\s*except.*:\s*$" /tmp/pr<N>.diff | grep -E "^\+\s*pass\s*$"
```

Bare `except:` swallows `KeyboardInterrupt` and `SystemExit`. `except Exception: pass` swallows everything else. Neither acceptable without a justifying comment. Fix: catch specific exceptions, log or re-raise.

### Hallucinated transformers symbol [B]

```bash
grep -hE "^\+from transformers[\.\w]* import" /tmp/pr<N>.diff | sed 's/,/\n/g' | grep -oE "[A-Za-z_][A-Za-z0-9_]*"
# Then for each symbol:
python -c "from transformers.models.<x> import <SYMBOL>"
```

LLMs sometimes invent class names (`Qwen3_5MoeFlashAttention2`, `LlamaForFastGeneration`, etc.). A nonexistent symbol crashes the library at import time. The pinned version is `transformers==5.5.4` per `pyproject.toml`. Fix: verify the symbol exists in the pinned version; if it doesn't, the import was hallucinated.

### Wildcard import in library [I]

```bash
grep -nE "^\+from QEfficient[\.\w]* import \*" /tmp/pr<N>.diff
```

`from X import *` in production code makes the public surface ambiguous and can conflict-rebind names. Fix: list the names explicitly.

### Imported but never used [I]

For new imports:

```bash
ruff check --select F401 --diff
```

Or grep each new import name in the file's body. In `pytorch_transforms.py` specifically: an HF or QEff class imported but absent from any mapping is dead weight. Fix: remove the import, or wire up the missing mapping the import was meant to support.

### Feature toggled by environment variable [I]

```bash
grep -nE "^\+.*os\.environ\.(get|setdefault)\(" /tmp/pr<N>.diff
grep -nE "^\+.*os\.environ\[" /tmp/pr<N>.diff
```

Public-API behavior selected by `os.environ.get("MY_FLAG")` instead of an explicit kwarg is anti-pattern in a library. Exempt: module-top-level reads of operational tunables that don't affect the exported graph (e.g. log level, worker count). Flag when an env-var read sits inside a public method body and selects a code path. Fix: add a kwarg with a default, document it, plumb it through hash_params.

### Class-level mutable state shared across instances [B]

```bash
grep -nE "^\+\s+_(start|end|total_layers|cur_layer|state|idx|batch_size)\s*=\s*[0-9None]" /tmp/pr<N>.diff
```

```python
class QEFFBaseModel(ABC):
    _start = 0
    _end = 1
    _total_layers = None
```

Class variables that are then mutated externally (`QEFFBaseModel._start = idx`) become global state shared across instances and threads — concurrent exports corrupt each other. Fix: pass through args/kwargs, instance state, or thread-local context.

### Toggle by commenting [B]

```bash
grep -nE "^\+\s*#.*(comment.*out|enable.*disable|toggle|uncomment)" /tmp/pr<N>.diff
```

```python
# qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
```

Examples that require the user to hand-edit source to switch modes are user-hostile. Fix: argparse flag.

### Change landed in the wrong layer [I → B]

The most common architectural slop in agent-authored PRs: model-specific logic added to a base/shared class because that's where the author was already editing, not where the change belongs.

```bash
# Base/shared files inherited by every model — scrutinize every hunk here:
grep -nE "^\+\+\+ b/(QEfficient/base/(modeling_qeff|common|pytorch_transforms|onnx_transforms)\.py|QEfficient/transformers/models/modeling_auto\.py|QEfficient/transformers/cache_utils\.py)" /tmp/pr<N>.diff

# Model-specific conditional / class name leaking into generic code:
grep -nE "^\+.*(model_type\s*==\s*[\"'][a-z0-9_]+[\"']|isinstance\([^,]+,\s*[A-Z][A-Za-z0-9_]*(Model|Config|ForCausalLM|ForConditionalGeneration)\))" /tmp/pr<N>.diff
```

Apply the decision tree from `repo_conventions.md` (`## Layering / change-placement contract`):

- Behavior for **one model** in a base class / Auto registry / `cache_utils.py` → **Blocker**. Move to `models/<model>/modeling_<model>.py`.
- Helper for **a few models** dumped in a base class → **Issue**. Move to `QEfficient/transformers/modeling_utils.py`.
- **Generation / decode-loop / sampling / runtime-KV / VLM-input** logic in `modeling_auto.py`'s `generate()` → **Issue**. Move to `QEfficient/generation/`.
- ONNX surgery → `QEfficient/exporter/`; compile orchestration → `QEfficient/compile/`. Misplaced → **Issue**.
- **Any** `if model_type == "<x>"`, `isinstance(self.model, <SpecificModel>)`, or concrete model/config name inside generic code → **Blocker** (the generic layer must not know which model it runs).
- A new method/kwarg/branch on a base class with no "all models need this" justification → **Issue**, → **Blocker** if it embeds a model name or quirk.

A legitimate generic lifecycle change (every Auto class uses it, no model name embedded) is fine — don't flag it. But a diff concentrated in base/shared files is a yellow flag: well-scoped model work lives in `models/<model>/`, `tests/`, `examples/`, `docs/` and adds only a few mapping/import lines to `pytorch_transforms.py` / `modeling_auto.py`. For each finding, name the file the code should live in.

### New QEff class not registered in `pytorch_transforms.py` [B]

Automated check (run from repo root):

```bash
comm -23 \
  <(grep -hE "^\+class QEff[A-Za-z0-9_]+" /tmp/pr<N>.diff \
    | sed -E 's/^\+class (QEff[A-Za-z0-9_]+).*/\1/' | sort -u) \
  <( (grep -hoE "QEff[A-Za-z0-9_]+" QEfficient/transformers/models/pytorch_transforms.py; \
       grep -hoE "QEff[A-Za-z0-9_]+" QEfficient/transformers/models/modeling_auto.py) \
     | sort -u)
```

Any name printed here is a `QEff*` class added by the PR but never referenced from `pytorch_transforms.py` or `modeling_auto.py` — likely dead code.

But not every QEff class lives in `_module_mapping`. Three legitimate registration paths:

1. `_module_mapping: Dict[type, type]` on a `ModuleMappingTransform` subclass.
2. `_match_string_replace_method` on an `ExternalModuleMapperTransform` subclass — this matches HF class **name strings**. Used by `KVCacheExternalModuleMapperTransform`, `PrefillOnlyExternalModuleMapperTransform`, `RevertPrefillOnlyExternalModuleMapperTransform` (e.g. DeepseekV3, Molmo, Grok1).
3. Explicit invocation from an Auto-class method — e.g. `prefill()` calls `PrefillOnlyExternalModuleMapperTransform.apply` directly.

A class found by the diff above could still be wired via path 2 or 3 — verify before declaring dead. Fix: add the appropriate mapping or confirm explicit invocation.

### Hash invariant break [B]

```bash
# New kwargs added to public methods:
grep -nE "^\+.*def (export|compile|from_pretrained)\(" /tmp/pr<N>.diff -A20

# Where hash params get plumbed:
grep -nE "^\+.*(hash_params\[|create_export_hash|hash_dict_params|compile_hash_params|KWARGS_INCLUSION_LIST)" /tmp/pr<N>.diff
```

For each new graph-affecting kwarg or compile flag, confirm one of:

- `self.hash_params[<key>] = <value>` — the dominant pattern; assigned in `__init__`, mutated under `transform`/`export`. See `QEfficient/transformers/models/modeling_auto.py` for examples.
- `KWARGS_INCLUSION_LIST` in `QEfficient/utils/constants.py` — for kwargs passed through `__init__`/`from_pretrained` and read by `create_model_params`.
- `compile_hash_params` dict in `QEfficient/base/modeling_qeff.py` (`_compile`) — for compile-time switches.

A kwarg invisible to the cache means users hit a stale QPC after switching configs and silently get wrong results. The cache key is SHA256[:16], computed by `create_export_hash` (export dir) and `hash_dict_params(compile_hash_params)` (compile dir). Mandatory fix.

Also flag: `hash_params[k] = str(some_dict)` — `str(dict)` order is insertion-order in CPython 3.7+ but stringifying nested dicts is fragile; use `json.dumps(d, sort_keys=True)`.

### PrefillOnly/Revert asymmetry [B]

If the PR adds a `PrefillOnlyTransform._module_mapping` or `PrefillOnlyChunkedTransform._module_mapping` entry, the inverse `RevertPrefillOnlyTransform` (or `RevertPrefillKeepAttentionTransform`) entry must also be present. `RevertPrefillOnlyTransform` auto-derives its map by inverting the chunked mapping, but `RevertPrefillKeepAttentionTransform` is hand-written; a missing entry silently leaks chunked-prefill forwards into the decode QPC (the disagg flow compiles the same module in place multiple times, so an unreverted transform persists). Fix: add both directions of the mapping.

### Cross-class state coupling [B]

Generic infrastructure (`QEFFBaseModel`, `QEffDynamicCache`, generic Auto-class methods, `cache_utils.py`) referencing model-specific class state is a coupling bug regardless of how it was authored:

```python
# in QEfficient/transformers/cache_utils.py (generic):
if cache.layer_types[Qwen3_5MoeTextModel._start] == "full_attention": ...
```

A grep for `<SpecificModel>\.[_A-Z]` in generic modules surfaces these. Fix: pass needed indices as args, or move the logic into the model-specific module.

### Misplaced helpers in `QEfficient/utils/` [I]

`QEfficient/utils/` is for thin helpers (constants, hashing, logging, generate-inputs). Multi-hundred-line orchestration modules belong in domain-specific dirs:

- ONNX surgery / pipelining → `QEfficient/exporter/`.
- Compile orchestration → `QEfficient/compile/`.
- Inference loops → `QEfficient/generation/`.

Fix: move the file, or split the file's helpers from its orchestration.

### Duplicate function definitions [I]

```bash
awk '/^\+\s*def /{print FILENAME ":" NR " " $0}' /tmp/pr<N>.diff | sort -k2 | uniq -f1 -d
```

Two same-named functions added in the same diff (often nested inside two methods, with identical bodies) is copy-paste slop. Fix: extract to a module-level helper.

### Destructive self-mutation in forward [B]

```bash
grep -nE '^\+\s*self\.[a-zA-Z_][a-zA-Z0-9_]*\s*(\+=|-=|\*=|/=|=\s*self\.)' /tmp/pr<N>.diff
```

Look for `self.<attr> -= X`, `self.<attr> += X`, `self.<attr> = self.<attr> - X` inside a `forward()`. The attr persists across calls; a second forward double-applies the mutation. The grep above intentionally excludes plain `self.x = <other_thing>` which is too noisy (legitimate field assignment uses the same shape). When a hit fires, manually verify it's inside `forward()` (or any per-call hot path). Fix: compute into a local, don't store back to `self`.

### Hardcoded compute dtype [I → B]

```bash
grep -nE '^\+.*dtype\s*=\s*torch\.(float32|get_default_dtype\(\))' /tmp/pr<N>.diff
```

QEff exports every model in FP32, FP16 **and** BF16. A tensor that feeds the graph (RoPE cos/sin cache, `get_dummy_inputs`, KV/attention buffers, mask fill, `IOInfo` dtype) with a hardcoded `dtype=torch.float32` / `torch.get_default_dtype()` forces full precision into a half-precision graph and breaks custom_io — silent precision corruption. Must derive from `config.torch_dtype` (or the relevant sub-config). **Exempt** the deliberate up-cast-then-restore idiom `softmax(..., dtype=torch.float32).to(query.dtype)` — verify manually. Fix: use `config.torch_dtype`. Issue → Blocker when it feeds attention scores or KV.

### Hardcoded masked-attention fill value [B]

```bash
grep -nE '^\+.*(masked_fill|torch\.where|torch\.full|torch\.tensor|fill_)\(.*-?(10000|1e4|1e9|50000|3\.0e4)' /tmp/pr<N>.diff
grep -nE '^\+.*torch\.finfo\([^)]*\)\.min' /tmp/pr<N>.diff
```

Masked-position fills must use `MIN_MASKED_ATTENTION_VALUE` (`QEfficient/utils/constants.py`, = `float("-inf")`), never a finite magic number (`-10000`, `-1e4`, `torch.finfo(...).min`) — a finite sentinel leaks probability mass through masked positions. And if the PR touches `FP16ClipTransform` / `_onnx_transforms`, confirm the `-inf`-preservation branch survives (clipping `-inf` to fp16-min silently un-masks). Fix: use the constant. Blocker.

### Config entry shadowed by a skip-list [B]

```bash
# For each new tests/configs entry, grep the model name in the skip-lists:
grep -nE '"<model_name>"' QEfficient/utils/test_utils.py   # SKIPPED_MODELS / FULL_MODEL_TESTS_TO_SKIP
```

A PR that adds a `tests/configs/*.json` entry **and** lists the same model in `ModelConfig.SKIPPED_MODELS` / `FULL_MODEL_TESTS_TO_SKIP` runs zero parity — the config presence games the "missing config = Blocker" check. Skip-listing is legitimate only for a known upstream HF-transformers defect, stated in the PR body. Fix: remove the skip, or justify it. Blocker.

### WIP / Draft PR proposed for merge [B]

```bash
gh pr view <N> --repo quic/efficient-transformers --json isDraft,title --jq '{isDraft, title}'
```

A PR in GitHub draft state, or whose title matches `(?i)\bWIP\b|\[draft\]|do[- ]not[- ]merge|\bDNM\b|test only`, is self-labeled incomplete and must not merge (a `[WIP]` PR has been merged-then-reverted here before). Fix: drop the marker and re-request review once ready. Blocker.

### Commented-out imports / code [I]

```bash
grep -nE "^\+\s*#+\s*(import |from |class |def |return |@ )" /tmp/pr<N>.diff
```

Half-done wiring (`# from X import Y`, commented map entries, commented-out function bodies) is noise unless paired with a comment explaining when to re-enable. Fix: delete or wire up.

### TODO/FIXME for things that should be done now [I]

```bash
grep -nE "^\+\s*#\s*(TODO|FIXME|HACK|XXX)" /tmp/pr<N>.diff
```

A TODO/FIXME pointing at follow-up work tracked elsewhere is fine. A TODO/FIXME describing work the *current* PR should have done is slop. Fix: handle now or open a ticket and reference it in the comment.

### Comments referencing the PR/review [I]

```bash
grep -nE "^\+\s*#\s*(Added (for|per)|Per review|Per feedback|Fix for [A-Z]+-[0-9]+|For ticket|This was the|Used by)" /tmp/pr<N>.diff
```

`# Added for issue #123`, `# Per review feedback`, `# Fix for PROJ-1234` belong in the PR body, not in source. They rot. Fix: delete.

### Comments restating the code [N]

```python
counter += 1  # Increment counter
```

Noise. Fix: delete.

### AI-style multi-paragraph docstrings on trivial getters [I]

Look for docstrings 4+ lines long on functions whose body is 1-2 lines — typical AI fill. The repo prefers terse or absent docstrings. A bad docstring is worse than no docstring. Fix: trim to one line or delete.

### Docstring args don't match signature [I]

Read each docstring's `Args:` block against the signature. If they disagree (renamed param, removed param still listed, added param not listed), the docstring lies. Fix: align or delete.

### Magic numbers in models / examples [I]

```bash
grep -nE "^\+\s*[A-Z_][A-Z0-9_]{3,}\s*=\s*[0-9]+\s*$" /tmp/pr<N>.diff
```

Hardcoded `CTX_LEN = 128` next to a model whose `config.max_position_embeddings = 4096` — either there's a reason and it needs a comment, or it's slop. Fix: take from `config.*`, or annotate the why.

### Examples importing private names [I]

```bash
grep -nE "^\+from QEfficient\.[^ ]+ import [^ ]*_[A-Za-z]" /tmp/pr<N>.diff | grep "^\+\+\+ b/examples/"
```

Examples must use the public surface in `QEfficient/__init__.py`. Importing `_QEffAutoModelForImageTextToTextDualQPC` or any leading-underscore symbol in an example couples to internals. Fix: switch to the public class.

### Hand-built `past_key_values` to test "parity" [B]

If a test does:

```python
qeff_inputs = _make_hand_built_pkv(transformed, ...)
out = transformed(**qeff_inputs)
```

it's not testing parity — it's testing that the model accepts the test's hand-rolled tuple shape. The cache the transform installs is the thing under test. Fix: use the production input path (`get_dummy_inputs` from the wrapper), or rely on the parametrized 4-stage parity tests.

### Smoke test masquerading as parity [I]

A test that asserts only `torch.isfinite(logits).all()` or `isinstance(transformed, QEff<X>)` is a smoke test. Useful, but never catches a logic bug. If smoke is the only test added for a new model, flag as low-value — the bar is the 4-stage parity from CONTRIBUTING.md.

### Test that fixes a parameter so the buggy path doesn't run [B]

A test parametrized with `n_group=1` makes the group-constrained top-k routing path a mathematical no-op (it degenerates to a flat global top-k), hiding a router divergence from HF for the real multi-group config. Look for: `n_group=1`, `num_experts_per_group=1`, `num_hidden_layers=1` for cross-layer bugs, single-token sequences for sequence-state bugs. Also watch for parity tests that compare QEff-vs-QEff (same router on both sides) instead of QEff-vs-unmodified-HF — those can never catch a divergence introduced in the QEff forward. Fix: add a config that exercises the buggy regime and assert against HF.

### Test monkey-patches the SUT [B]

```bash
grep -nB2 -A4 "monkeypatch\.setattr|@patch\(['\"]QEfficient" /tmp/pr<N>.diff
```

When a test patches `QEfficient.X.Y` and then calls `Y` to assert it works — the patched target is the SUT, so the test only checks the patch executes. Fix: test the production path or test the patch's own contract, not both pretending to be one.

### `validate.md` row that is just TBD / — / ? [B]

```bash
grep -nE "^\+.*\|\s*(TBD|—|--|\?|N/A)\s*\|" docs/source/validate.md
```

A "documented" row that says `TBD` is not compliance — it's gaming the check. Fix: fill the row or remove it.

### Trailing whitespace lines [N → I if widespread]

```bash
grep -cE "^\+.*[[:space:]]$" /tmp/pr<N>.diff
```

Pre-commit normally strips trailing whitespace. >5 trailing-WS adds means the author bypassed pre-commit (`--no-verify`) — the bypass itself is the signal. Fix: re-run pre-commit and re-stage.

### Imports out of isort order [N]

In a new `.py`: stdlib → third-party → first-party (`QEfficient`), each group alphabetized. Pre-commit enforces; manual edits frequently break it. Fix: `ruff check --fix`.

### Lint suppression added with feature code [I]

```bash
grep -nE "^\+.*\b(noqa|type:\s*ignore|pylint:|fmt: off)" /tmp/pr<N>.diff
```

A new feature PR shipping `# noqa: F401` / `# type: ignore` is the author silencing a real defect rather than fixing it. Fix: address the underlying issue.

### `assert` without message [N → I in error paths]

```bash
grep -nE "^\+\s*assert\s+[^,]+\s*$" /tmp/pr<N>.diff
```

`assert prefill_seq_len > 1` with no message — when it fails the user gets a stack with no clue. Fix: add a message string.

### Mass cosmetic changes mixed with feature work [I]

A PR titled "Add support for X" that also rewrites unrelated docstrings, renames variables in untouched modules, or reformats whole files should be split. Fix: ask the author to separate refactor from feature.

### Duplicated example files [I]

```bash
# For two new examples in the same task dir:
diff examples/<task>/models/<m1>/foo.py examples/<task>/models/<m2>/foo.py
```

If two new example files differ only by model name and a couple of constants, they're duplication slop. Fix: parameterize one example, or extract a shared helper under `examples/<task>/_helpers.py`.

### Empty PR description / vague title [N → I if PR is large]

```bash
gh pr view <N> --json body,title,additions,changedFiles --jq '{title, body_len: (.body | length), adds: .additions, files: .changedFiles}'
```

If `body == ""` AND `additions ≥ 1000` OR `changedFiles ≥ 10` → escalate to **I** (large PRs without description are either AI-only authored or undisciplined; either way they cost reviewer time). Fix: ask the author to write a description.

### Many "lint fix" / "fix CI" commits [N]

```bash
gh pr view <N> --json commits --jq '.commits[].messageHeadline'
```

5+ "lint fix" / "fix CI" / "address comment" commits in a row signal undisciplined development. Squash on merge.

### Oversized or multi-responsibility method [I → B]

Ruff caps line length, not method length. The structural PEP 8 rule: new methods ≤ ~50 lines, ≤ ~80 lines for a genuinely linear sequence; 100 lines without internal helpers = **Issue**; > 200 lines OR ≥ 3 mixed responsibilities (arg-validate + reshape + cache update + ONNX prep in one body) = **Blocker** — unreviewable, almost always hiding a bug. Use the audit grep to list every added `def`, then count lines of each in the file:

```bash
grep -nE '^\+\s*(async )?def [a-zA-Z_]' /tmp/pr<N>.diff
```

Fix: propose split points (preprocessing / core / postprocessing, per-branch helpers). The 100-line Auto-class methods are legacy *upper bound*, not a target.

### Deep nesting / cognitive complexity [I → B]

The PEP-8 rule ruff can't enforce. Severity by structure:

- Nesting depth 3 (`if` inside `for` inside `try`) → **Issue**; ask for an early return / guard / extracted helper.
- Nesting depth ≥ 4, OR > 5 boolean operators in one condition, OR a `for`/`while` body > 30 lines → **Blocker**.
- Mutating shared state inside a deeply nested branch (`self.x = ...` four levels in) → **Blocker** — mutation should be at one named site.
- A `forward()` whose flow can't be summarized in one sentence ("project, attend, project, residual") → **Issue**.

Heuristic to surface candidates (then read the function):

```bash
grep -nE '^\+\s{16,}' /tmp/pr<N>.diff      # ≥ 4 indent levels (4-space style)
grep -nE '^\+.*(\sand\s.*){4,}|(\sor\s.*){4,}' /tmp/pr<N>.diff   # > 4 boolean joins
```

Fix: extract a helper, invert the condition for an early return, or split the loop body.

### Misleading or generic names [I → B]

Mechanical case violations are **Nit**. Semantic naming problems are not:

- Generic placeholders in non-trivial code (`result`, `data`, `tmp`, `temp`, `helper`, `obj`, `thing`, `value`, `info`, `output2`, `processed_data`, `do_stuff`, `handle`, `process`) → **Issue**. Describe what flows through (`attn_scores`, `routed_logits`, `kv_block_indices`).
- **Misleading** names (a `mask` that's actually `position_ids`; a `get_*` that mutates; an `_apply_*` that returns unchanged; an `enable_*` flag that disables) → **Blocker** — they cause bugs at the call site.
- QEff wrapper naming: new attention/layer/model wrappers must follow `QEff<HFClassName>[<Suffix>]`. `QeffLLamaAttn` / `Qeff_Llama_Attention` → **Issue** (breaks the spelling transform registration matches against).
- Booleans: use `is_*` / `has_*` / `should_*` / `use_*`. A bool named `prefill_or_decode` or `cb` → **Nit**, ask for `is_prefill` / `use_continuous_batching`.

Fix: rename. Mechanical greps can't catch semantic misleading names — read the variable's actual usage at the call site.

### Mutable default arg [B]

```bash
grep -nE '^\+\s*def [a-zA-Z_][a-zA-Z0-9_]*\([^)]*=\s*(\[\]|\{\}|set\(\))' /tmp/pr<N>.diff
```

Classic Python footgun: `def f(x=[])` / `x={}` — the default is shared across every call. Fix: use `None` and create inside the body.