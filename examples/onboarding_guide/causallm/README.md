# Onboarding a CausalLM Model

## Prerequisites

Install `qefficient-transformers` library in editable mode:
```sh
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
pip install -e .
```

---

## Transformers Version Compatibility

**Important:** QEfficient has a pinned `transformers` library version dependency.

**Check the current version:**
```bash
grep "transformers==" pyproject.toml
```

See `dependencies` in [`pyproject.toml`](../../../pyproject.toml) for the exact version.

**Compatibility rules:**
- You can only onboard models that are supported in the pinned transformers version or earlier
- Models added to transformers after this version are not yet supported
- Always verify when your target model was added to the transformers library

**How to verify model compatibility:**

1. Check transformers release history at [HuggingFace Transformers Releases](https://github.com/huggingface/transformers/releases)
2. Find the release where your model was first introduced
3. Compare versions:
   - If model's release version ≤ QEfficient's pinned version → Proceed with onboarding
   - If model's release version > QEfficient's pinned version → Cannot onboard yet


**Need a newer model?**

If you need to onboard a model that requires a newer transformers version:
1. Open an issue on the [QEfficient GitHub repository](https://github.com/quic/efficient-transformers/issues)
2. Request a transformers version bump
3. Provide justification and the specific model you need

---

## Introduction

This guide walks you through onboarding a new CausalLM model to QEfficient-transformers. We use an example model named `Blueprint` to demonstrate the required changes.

---

## Onboarding Process

```mermaid
flowchart TD
    A["Check Transformers Library
• Locate model in transformers/models/&lt;model&gt;/modeling_*.py
• Identify architecture classes (Attention, DecoderLayer, etc.)"]
    
    B{"Class already
Implemented"}
    
    C["Create Custom Files
• Create modeling_*.py
• Implement custom classes
• Add __qeff_init__ methods"]
    
    D["Test the model using
the auto model class
and validate the
functionality"]
    
    E["Add Mappings in pytorch_transforms.py
• CustomOpsTransform (RMSNorm)
• KVCacheTransform (all model classes)
• ExternalModuleMapperTransform (if needed)"]
    
    K{"if all test passes"}
    
    L["Debug & Fix Issues
Retest with test pipelines"]
    
    M["Submit PR
(Follow
CONTRIBUTING
guidelines)"]
    
    A --> B
    B -->|No| C
    B -->|Yes| D
    C --> E
    E --> F
    
    subgraph F["Testing Pipeline (4 Stages)"]
        direction TB
        G["Stage 1: PyTorch HF Model (Baseline)
(tokens should match)"]
        H["Stage 2: PyTorch KV Model (After QEff transforms)
(tokens should match)"]
        I["Stage 3: ONNX/ORT Model (After export)
(tokens should match)"]
        J["Stage 4: Cloud AI 100 (Hardware execution)
(tokens should match)"]
        
        G --> H
        H --> I
        I --> J
    end
    
    F --> K
    K -->|No| L
    L --> F
    K -->|Yes| M
```

---

## Step 1: Check Transformers Library

1. **Locate the model** in the transformers library:
   - Path: `/src/transformers/models/<model_name>/modeling_<model_name>.py`
   - Example: `/src/transformers/models/blueprint/modeling_blueprint.py`

2. **Identify required classes**:
   - Attention Layer
   - Decoder Layer
   - Model (main class)
   - ForCausalLM (top-level)
   - RMSNorm/LayerNorm
   - RotaryEmbedding (if applicable)

3. **Check existing implementations** in `QEfficient/transformers/models/`:
   - If similar classes exist → Reuse patterns
   - If not → Create custom implementations

---

## Step 2: Create Custom Files & Mappings

### 2.1 Create Custom Modeling File

Create directory structure:
```
QEfficient/transformers/models/blueprint/
├── __init__.py
└── modeling_blueprint.py
```

**Key modifications in `modeling_blueprint.py`:**
- `QEffBlueprintRotaryEmbedding`: Precompute sin/cos for rotary embeddings
- `QEffBlueprintAttention`: Use `position_ids`, return `past_key_value`, implement `__qeff_init__`
- `QEffBlueprintDecoderLayer`: Return `past_key_value` from forward pass
- `QEffBlueprintModel`: Use `QEffDynamicCache` instead of standard cache
- `QEffBlueprintForCausalLM`: Entry point with additional parameters

See `modeling_example.py` for detailed implementation examples.

### 2.2 Add Mappings in pytorch_transforms.py

**CustomOpsTransform** (RMSNorm mapping):
```python
class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        BlueprintRMSNorm: CustomRMSNormAIC,
    }
```

**KVCacheTransform** (all model classes):
```python
class KVCacheTransform(ModuleMappingTransform):
    _module_mapping = {
        BlueprintAttention: QEffBlueprintAttention,
        BlueprintDecoderLayer: QEffBlueprintDecoderLayer,
        BlueprintModel: QEffBlueprintModel,
        BlueprintForCausalLM: QEffBlueprintForCausalLM,
    }
```

See `example_pytorch_transforms.py` for complete example.

---

## Step 3: Testing (4-Stage Pipeline)

Your implementation is validated through four stages:

| Stage | Description | Validation |
|-------|-------------|------------|
| **1. PyTorch HF** | Original transformers model | Baseline tokens |
| **2. PyTorch KV** | After QEff transforms | Tokens match Stage 1 |
| **3. ONNX/ORT** | After export to ONNX | Tokens match Stage 2 |
| **4. Cloud AI 100** | Hardware execution | Tokens match Stage 3 |

**Test function:** `check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100` in `tests/transformers/models/test_causal_lm_models.py`

### Common Issues

**Token mismatch (Stage 1→2):**
- Check all classes are mapped in `KVCacheTransform`
- Verify `__qeff_init__` methods exist
- Ensure `position_ids` are correctly passed

**ONNX export failure (Stage 2→3):**
- Check for unsupported PyTorch operations
- Verify dynamic shapes are defined

**Compilation failure (Stage 3→4):**
- Reduce `num_cores` or model size
- Leave `device_id` / `device_ids` unset unless a test or example must pin a specific QAIC group.

---

## Step 4: Add to Test Suite

Edit `tests/transformers/models/test_causal_lm_models.py`:

```python
test_models_causal = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    # ... existing models ...
    "YourOrg/YourModel-7B",  # Add your model here
]
```

**Run tests:**
```bash
# Test your specific model
pytest tests/transformers/models/test_causal_lm_models.py::test_custom_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100 -k "YourModel" -v

# Run all regular tests
pytest tests/transformers/models/test_causal_lm_models.py -m regular
```

---

## Step 5: Validation Checklist

Before submitting PR:

**Implementation:**
- [ ] Created `QEfficient/transformers/models/<model_name>/` directory
- [ ] Implemented all required custom classes
- [ ] Added mappings in `CustomOpsTransform` and `KVCacheTransform`
- [ ] Added imports at top of `pytorch_transforms.py`

**Testing:**
- [ ] Model added to `test_models_causal` list
- [ ] All 4 stages pass (PyTorch HF → KV → ORT → AI 100)
- [ ] Continuous batching tests pass
- [ ] `qconfig.json` generated successfully

**Code Quality:**
- [ ] Code follows project style guidelines
- [ ] Commits use DCO sign-off (`git commit -s`)
- [ ] Branch created from `main`

---

## Step 6: Submit Pull Request

Follow guidelines in [CONTRIBUTING.md](../../../CONTRIBUTING.md):

1. Create feature branch: `git checkout -b add-yourmodel-support main`
2. Commit with DCO: `git commit -s -m "Add support for YourModel"`
3. Push and create PR targeting `main` branch
4. Include test results in PR description

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Token mismatch between stages | Check class mappings, verify `position_ids` handling |
| Shape errors | Verify KV cache dimensions, check `past_key_value` returns |
| ONNX export fails | Replace unsupported ops, define dynamic shapes |
| Compilation fails | Reduce `num_cores`, check device availability |
| Runtime errors | Verify input shapes match specializations |

**Debug tip:** Start with `n_layer=1` and short prompts, then gradually increase complexity.

---

## References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [QEfficient Transformers](https://github.com/quic/efficient-transformers)
- [Contributing Guidelines](../../../CONTRIBUTING.md)
- [Test Suite](../../../tests/transformers/models/test_causal_lm_models.py)
