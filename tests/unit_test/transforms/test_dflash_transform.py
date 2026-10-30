# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the DFlash transforms in QEfficient.transformers.models.pytorch_transforms
(DFlashTransform, DFlashDLMTransform, DFlashTLMTransform) and for
QEfficient.transformers.spd.dflash.compute_dflash_target_hidden_states.

Verifies that:
  1. DFlashTransform swaps QEff Qwen3 modules for their DFlash draft-model (QEffDFlash*) counterparts.
  2. DFlashDLMTransform cleanly drops fc/hidden_norm on the draft side, with no checkpoint required.
  3. DFlashTLMTransform attaches fc/hidden_norm/target_layer_ids on the target side, for multiple
     model families, with no checkpoint required.
  4. The DFlash draft model's forward pass (two-stream rotary embedding, two-stream attention,
     custom causal mask) runs correctly on a tiny in-memory model.
  5. The target-model side's target_layer_ids collection + compute_dflash_target_hidden_states
     wiring produces the expected hidden_states shape end to end.
  6. compute_dflash_target_hidden_states itself matches a manual reference computation.

All tests run on CPU only, using tiny in-memory models -- no checkpoints, no Hugging Face Hub access.
"""

import copy

import pytest
import torch
from torch import nn

VOCAB_SIZE = 500
CTX_LEN = 32
HIDDEN_SIZE = 64


def make_tiny_qwen3_5_vlm(moe=False):
    from transformers import (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
    )

    config_cls = Qwen3_5MoeConfig if moe else Qwen3_5Config
    model_cls = Qwen3_5MoeForConditionalGeneration if moe else Qwen3_5ForConditionalGeneration
    config = config_cls(
        text_config={
            "vocab_size": VOCAB_SIZE,
            "hidden_size": 32,
            "intermediate_size": 64,
            **(
                {
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32,
                    "shared_expert_intermediate_size": 32,
                }
                if moe
                else {}
            ),
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "max_position_embeddings": CTX_LEN,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    config._attn_implementation = "eager"
    return model_cls(config).eval()


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("seq_len,num_logits", [(32, 16), (128, 1)])
def test_qwen3_5_dflash_vllm_returns_vocabulary_logits(moe, seq_len, num_logits):
    from QEfficient import QEFFAutoModelForImageTextToText

    torch.manual_seed(23)
    hf_model = make_tiny_qwen3_5_vlm(moe=moe)
    hf_model.config.text_config.max_position_embeddings = 256
    qaic_config = {"target_layer_ids": [1, 3], "speculative_model_type": "target"}
    qeff = QEFFAutoModelForImageTextToText(hf_model, kv_offload=True, qaic_config=qaic_config)
    if moe:
        qeff.transform(
            ctx_len=256,
            seq_len=128,
            bs=1,
            num_devices=1,
            aic_num_cores=4,
            qaic_config=qaic_config,
            prefill_only=True,
            prefill_seq_len=128,
            enable_chunking=True,
        )
    wrapper = qeff.lang_model.model
    inputs = qeff.model.get_dummy_inputs(kv_offload=True, batch_size=1)["lang"]
    inputs["input_ids"] = torch.randint(10, VOCAB_SIZE, (1, seq_len))
    inputs["input_ids"][:, :2] = hf_model.config.image_token_id
    inputs["vision_embeds"].normal_()
    inputs["position_ids"] = torch.arange(seq_len).view(1, 1, -1).repeat(4, 1, 1)
    inputs["num_logits_to_keep"] = torch.zeros((num_logits, 1), dtype=torch.int64)
    for index, kind in enumerate(hf_model.config.text_config.layer_types):
        if kind == "full_attention":
            inputs["past_key_values"][index] = [
                torch.zeros((*state.shape[:2], 256, state.shape[-1])) for state in inputs["past_key_values"][index]
            ]
    with torch.no_grad():
        actual = wrapper(**copy.deepcopy(inputs))
        wrapper.qaic_config = {"target_layer_ids": [1, 3]}
        greedy = wrapper(**copy.deepcopy(inputs))
    assert actual[0].shape == (1, num_logits, VOCAB_SIZE)
    assert torch.isfinite(actual[0]).all()
    torch.testing.assert_close(greedy[0], actual[0].argmax(-1).to(torch.int32))
    assert actual[3].shape == (1, seq_len, 32)
    torch.testing.assert_close(actual[1:], greedy[1:])
    assert qeff.lang_model.hash_params["dflash_output_logits"]


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("prompt_len", [1, 5])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("matrix_solve", [False, True])
@pytest.mark.parametrize("moe_flavour", [None, "decode_bmm", "expert_parallel"])
def test_qwen3_5_dflash_concatenated_matches_autoregressive(
    accepted, prompt_len, chunk_size, matrix_solve, moe_flavour, monkeypatch
):
    """Commit the prior prefix and verify the next block in one target call."""
    from QEfficient import QEFFAutoModelForImageTextToText

    torch.manual_seed(7)
    hf_model = make_tiny_qwen3_5_vlm(moe=moe_flavour is not None)
    hf_model.config.text_config.max_position_embeddings = 512
    reference = copy.deepcopy(hf_model).model.language_model
    qaic_config = {"target_layer_ids": [1, 3]}
    if moe_flavour:
        qaic_config["moe_config"] = {"flavour": moe_flavour}
    qeff = QEFFAutoModelForImageTextToText(hf_model, kv_offload=True, qaic_config=qaic_config)
    if moe_flavour:
        qeff.transform(
            ctx_len=512,
            seq_len=2 * chunk_size,
            bs=1,
            num_devices=1,
            aic_num_cores=4,
            qaic_config=qaic_config,
            prefill_only=True,
            prefill_seq_len=2 * chunk_size,
        )
    model = qeff.lang_model.model.language_model
    model.dflash_chunk_size = chunk_size
    assert qeff.lang_model.hash_params["dflash_cache_protocol"].endswith("fused_replay_v1")
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.dflash_matrix_solve = matrix_solve

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("DFlash must not compute the one-token recurrent branch")

    model_type = "qwen3_5_moe" if moe_flavour else "qwen3_5"
    monkeypatch.setattr(
        f"QEfficient.transformers.models.{model_type}.modeling_{model_type}.recurrent_gdn_decode_forward",
        unexpected_decode,
    )
    cache = qeff.model.get_dummy_inputs(kv_offload=True, ctx_len=512)["lang"]["past_key_values"]
    prompt = torch.randint(10, VOCAB_SIZE, (1, 2 * chunk_size))
    positions = torch.arange(2 * chunk_size).view(1, -1)
    positions[:, prompt_len:] = -1
    with torch.no_grad():
        prefill = model(
            input_ids=prompt,
            position_ids=positions,
            past_key_values=cache,
            cache_commit=torch.ones(1, dtype=torch.long),
            use_cache=True,
        )
        cache = prefill.past_key_values
        replay_ids = torch.zeros((1, chunk_size), dtype=torch.long)
        replay_positions = torch.full_like(replay_ids, -1)
        history = prompt[:, :prompt_len]
        for count in [accepted + 1, 1, chunk_size, 3]:
            ids = torch.randint(10, VOCAB_SIZE, (1, chunk_size))
            pos = torch.arange(history.shape[1], history.shape[1] + chunk_size).view(1, -1)
            output = model(
                input_ids=torch.cat([replay_ids, ids], dim=1),
                position_ids=torch.cat([replay_positions, pos], dim=1),
                past_key_values=cache,
                cache_commit=torch.zeros(1, dtype=torch.long),
                use_cache=True,
            )
            expected = reference(input_ids=torch.cat([history, ids], dim=1), use_cache=False, output_hidden_states=True)
            torch.testing.assert_close(
                output.last_hidden_state[:, -chunk_size:],
                expected.last_hidden_state[:, -chunk_size:],
                rtol=2e-4,
                atol=2e-4,
            )
            projection = model.hidden_norm(
                model.fc(torch.cat([expected.hidden_states[i][:, -chunk_size:] for i in [1, 3]], dim=-1))
            )
            torch.testing.assert_close(output.hidden_states[:, -chunk_size:], projection, rtol=2e-4, atol=2e-4)
            if torch.all(replay_positions == -1):
                for layer in (0, 2):
                    torch.testing.assert_close(output.past_key_values[layer], cache[layer], rtol=0, atol=0)
            cache = output.past_key_values
            replay_ids, replay_positions = ids, pos.clone()
            replay_positions[:, count:] = -1
            history = torch.cat([history, ids[:, :count]], dim=1)
        restarted = model(
            input_ids=prompt,
            position_ids=positions,
            past_key_values=cache,
            cache_commit=torch.ones(1, dtype=torch.long),
            use_cache=True,
        )
        for layer in (0, 2):
            torch.testing.assert_close(
                restarted.past_key_values[layer], prefill.past_key_values[layer], rtol=2e-4, atol=2e-4
            )


@pytest.mark.parametrize(
    "moe_config", [None, {"cores_per_expert": 2, "tree_reduce": True, "expert_parallel_chunk_size": 16}]
)
def test_qwen3_5_moe_dflash_preserves_upstream_expert_parallel(moe_config):
    from transformers import Qwen3_5MoeForConditionalGeneration

    from QEfficient import QEFFAutoModelForImageTextToText
    from QEfficient.transformers.moe import MoEFlavour
    from QEfficient.transformers.moe.block import QEffMoEBlockMixin

    torch.manual_seed(19)
    config = make_tiny_qwen3_5_vlm(moe=True).config
    config.text_config.num_experts = 256
    config.text_config.num_experts_per_tok = 8
    reference = Qwen3_5MoeForConditionalGeneration(config).eval()
    qaic_config = {"target_layer_ids": [1, 3]}
    if moe_config is not None:
        qaic_config["moe_config"] = moe_config
    qeff = QEFFAutoModelForImageTextToText(copy.deepcopy(reference), kv_offload=True, qaic_config=qaic_config)
    qeff.transform(
        ctx_len=256,
        seq_len=128,
        bs=1,
        num_devices=4,
        aic_num_cores=8,
        qaic_config=qaic_config,
        prefill_only=True,
        prefill_seq_len=128,
    )
    options = moe_config or {}
    cores_per_expert = options.get("cores_per_expert", 1)
    packed_chunks = 8 if moe_config is not None else 1
    language_model = qeff.lang_model.model.language_model
    assert language_model.target_layer_ids == [1, 3]
    for layer in language_model.layers:
        block = layer.mlp
        assert block.forward.__func__ is QEffMoEBlockMixin.forward
        assert block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
        assert block.num_devices == 4 and block.total_avl_cores == 32
        assert block.num_parallelized_experts == 32 // cores_per_expert
        assert block.num_pipeline_stages == 8 * cores_per_expert
        assert block.tree_reduce is options.get("tree_reduce", False)
        assert block.expert_parallel_num_packed_chunks == packed_chunks
    assert qeff.lang_model.hash_params["moe_prefill_flavour"] == "expert_parallel"
    assert qeff.lang_model.hash_params["moe_prefill_num_packed_chunks"] == packed_chunks

    # Check routed and shared experts at both verification and prompt-prefill lengths.
    with torch.no_grad():
        for seq_len in (16, 128):
            hidden = torch.randn(1, seq_len, config.text_config.hidden_size)
            actual = language_model.layers[0].mlp(hidden)
            expected = reference.model.language_model.layers[0].mlp(hidden)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_qwen3_5_dflash_requires_explicit_commit():
    from QEfficient import QEFFAutoModelForImageTextToText

    qeff = QEFFAutoModelForImageTextToText(
        make_tiny_qwen3_5_vlm(), kv_offload=True, qaic_config={"target_layer_ids": [1, 3]}
    )
    inputs = qeff.model.get_dummy_inputs(kv_offload=True)["lang"]
    inputs.pop("cache_commit")
    with pytest.raises(ValueError, match="requires cache_commit"):
        qeff.lang_model.model(**inputs)


@pytest.mark.parametrize("prefill_seq_len", [1, 15, 65])
def test_qwen3_5_dflash_rejects_unsupported_prefill(prefill_seq_len):
    from QEfficient import QEFFAutoModelForImageTextToText

    qeff = QEFFAutoModelForImageTextToText(
        make_tiny_qwen3_5_vlm(), kv_offload=True, qaic_config={"target_layer_ids": [1, 3]}
    )
    with pytest.raises(ValueError, match="positive multiple of 16"):
        qeff.model.get_specializations(batch_size=1, prefill_seq_len=prefill_seq_len, ctx_len=256, kv_offload=True)
    if prefill_seq_len < 2:
        with pytest.raises(ValueError, match="prefill_seq_len >= 2"):
            qeff.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=prefill_seq_len)


@pytest.mark.parametrize("dummy_seq_len", [16, 32, 128, 256, 1024])
@pytest.mark.parametrize("moe", [False, True])
def test_qwen3_5_dflash_export_traces_decode_width(dummy_seq_len, moe):
    from QEfficient import QEFFAutoModelForImageTextToText

    qeff = QEFFAutoModelForImageTextToText(
        make_tiny_qwen3_5_vlm(moe=moe),
        kv_offload=True,
        qaic_config={"target_layer_ids": [1, 3]},
    )
    decode_width = 32
    inputs = qeff.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=dummy_seq_len)["lang"]
    assert inputs["input_ids"].shape[1] == decode_width
    assert inputs["position_ids"].shape[-1] == decode_width
    if dummy_seq_len < 32:
        with pytest.raises(ValueError, match="prefill_seq_len >= 2"):
            qeff.model.get_specializations(batch_size=1, prefill_seq_len=dummy_seq_len, ctx_len=2048, kv_offload=True)
        return
    specs, _ = qeff.model.get_specializations(
        batch_size=1, prefill_seq_len=dummy_seq_len, ctx_len=2048, kv_offload=True
    )
    assert [(spec["seq_len"], spec["num_logits_to_keep"]) for spec in specs["lang"]] == [
        (dummy_seq_len, 1),
        (decode_width, 16),
    ]
    specs, _ = qeff.model.get_specializations(
        batch_size=1,
        prefill_seq_len=dummy_seq_len,
        ctx_len=2048,
        kv_offload=True,
        comp_ctx_lengths_prefill=[1024, 2048],
        comp_ctx_lengths_decode=[512, 1024, 2048],
    )
    assert [(spec["seq_len"], spec["num_logits_to_keep"]) for spec in specs["lang"]] == [
        (dummy_seq_len, 1),
        (dummy_seq_len, 1),
        (decode_width, 16),
        (decode_width, 16),
        (decode_width, 16),
    ]


@pytest.mark.parametrize("decode_width", [16, 32])
def test_qwen3_5_dflash_runtime_distinguishes_prefill_from_verify(decode_width):
    from types import SimpleNamespace

    from QEfficient.generation.dflash_generation import get_qwen35_target_seq_len

    shapes = [[(8, (1, 32)), (8, (1, 1))], [(8, (1, decode_width)), (8, (16, 1))]]
    session = SimpleNamespace(binding_index_map={"input_ids": 0, "num_logits_to_keep": 1}, allowed_shapes=shapes)
    if decode_width == 32:
        assert get_qwen35_target_seq_len(session, 16) == 32
        session.allowed_shapes = []
        session.bindings = [SimpleNamespace(dims=(1, 32)), SimpleNamespace(dims=(16, 1))]
        assert get_qwen35_target_seq_len(session, 16) == 32
    else:
        with pytest.raises(ValueError, match="Older two-call QPCs are unsupported"):
            get_qwen35_target_seq_len(session, 16)
    session.allowed_shapes = shapes[:1]
    with pytest.raises(ValueError, match="Re-export and recompile"):
        get_qwen35_target_seq_len(session, 16)


@pytest.mark.parametrize("chunk_size", [0, 1, 3, 24])
def test_qwen3_5_dflash_matrix_solve_requires_power_of_two(chunk_size):
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5GatedDeltaNet

    with pytest.raises(ValueError, match="power-of-two"):
        QEffQwen3_5GatedDeltaNet._dflash_triangular_inverse(None, None, chunk_size)


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


_TLM_MODEL_FACTORIES = [
    (make_tiny_qwen3, "qwen3"),
    (make_tiny_llama, "llama"),
]
_TLM_MODEL_IDS = [label for _, label in _TLM_MODEL_FACTORIES]


def _make_legacy_past_key_values(config, batch, ctx_len):
    """Zero-tensor legacy KV cache tuple, sized to ctx_len (mirrors test_blocking_transform.py)."""
    n_layers = config.num_hidden_layers
    n_attn = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_attn)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    return tuple(
        (
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )


# ---------------------------------------------------------------------------
# Tests: transform ordering
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.parametrize(
    ("qaic_config", "expected_order"),
    [
        ({"dflash_dlm": True}, ["dflash", "dflash_dlm", "spd", "sampler"]),
        ({"target_layer_ids": [1]}, ["dflash_tlm", "spd", "sampler"]),
    ],
)
def test_dflash_transforms_run_before_spd_and_sampler(monkeypatch, qaic_config, expected_order):
    from QEfficient import QEFFAutoModelForCausalLM
    from QEfficient.transformers.models.pytorch_transforms import (
        DFlashDLMTransform,
        DFlashTLMTransform,
        DFlashTransform,
        SamplerTransform,
        SpDTransform,
    )

    applied = []

    def record_apply(name):
        def apply(cls, model, config=None, **kwargs):
            applied.append(name)
            return model, False

        return classmethod(apply)

    monkeypatch.setattr(DFlashTransform, "apply", record_apply("dflash"))
    monkeypatch.setattr(DFlashDLMTransform, "apply", record_apply("dflash_dlm"))
    monkeypatch.setattr(DFlashTLMTransform, "apply", record_apply("dflash_tlm"))
    monkeypatch.setattr(SpDTransform, "apply", record_apply("spd"))
    monkeypatch.setattr(SamplerTransform, "apply", record_apply("sampler"))

    QEFFAutoModelForCausalLM(make_tiny_qwen3(), qaic_config=qaic_config)

    assert applied == expected_order


# ---------------------------------------------------------------------------
# Tests: draft (DLM) side -- DFlashTransform + DFlashDLMTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestDFlashDraftTransform:
    """DFlashTransform + DFlashDLMTransform on the draft (DLM) side -- Qwen3 only."""

    def _build_dlm(self):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_tiny_qwen3()
        return QEFFAutoModelForCausalLM(hf_model, qaic_config={"dflash_dlm": True})

    def test_unsupported_dlm_architecture_raises(self):
        from QEfficient import QEFFAutoModelForCausalLM

        with pytest.raises(NotImplementedError, match="QEffLlamaForCausalLM"):
            QEFFAutoModelForCausalLM(make_tiny_llama(), qaic_config={"dflash_dlm": True})

    def test_dflash_transform_swaps_to_dflash_classes(self):
        qeff_model = self._build_dlm()

        assert qeff_model.dflash_dlm is True
        assert type(qeff_model.model.model).__name__ == "QEffDFlashModel"
        assert type(qeff_model.model).__name__ == "QEffDFlashForCausalLM"
        for layer in qeff_model.model.model.layers:
            assert type(layer).__name__ == "QEffDFlashDecoderLayer"
            assert type(layer.self_attn).__name__ == "QEffDFlashAttention"

    def test_dflash_dlm_transform_clean_without_tlm_repo(self):
        qeff_model = self._build_dlm()

        assert not hasattr(qeff_model.model.model, "fc")
        assert not hasattr(qeff_model.model.model, "hidden_norm")

    def test_dflash_draft_forward_smoke(self):
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        qeff_model = self._build_dlm()
        model = qeff_model.model  # QEffDFlashForCausalLM

        block_size = 4
        batch = 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, block_size))
        target_hidden = torch.randn(batch, block_size, HIDDEN_SIZE)
        position_ids_target = torch.arange(block_size).unsqueeze(0)
        position_ids = position_ids_target + block_size
        attention_mask = torch.zeros(batch, 2 * block_size)  # only .shape[-1] is read by _create_mask
        # KV cache buffer must be pre-allocated to cover both the context block (positions
        # 0..block_size-1) and the noise block (positions block_size..2*block_size-1) that
        # get written into it -- an empty/lazy cache has nothing to index into at position 4+.
        legacy_pkv = _make_legacy_past_key_values(model.config, batch, ctx_len=2 * block_size)
        past_key_values = QEffDynamicCache.from_legacy_cache(legacy_pkv)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                target_hidden=target_hidden,
                position_ids=position_ids,
                position_ids_target=position_ids_target,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        assert outputs.logits.shape == (batch, block_size, VOCAB_SIZE)
        assert torch.isfinite(outputs.logits).all()

    @pytest.mark.parametrize("is_causal", [None, False, True])
    def test_mixed_attention_masks_follow_draft_config(self, is_causal):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        hf_model = make_tiny_qwen3()
        hf_model.config.layer_types = ["sliding_attention", "full_attention"]
        hf_model.config.sliding_window = 8
        hf_model.config.is_causal = is_causal
        qeff = QEFFAutoModelForCausalLM(hf_model, qaic_config={"dflash_dlm": True})
        assert qeff.hash_params["dflash_attention"] == "per_layer_window_v1"
        masks = []

        def record_mask(module, args, kwargs):
            masks.append(kwargs["attention_mask"].clone())

        handles = [
            layer.self_attn.register_forward_pre_hook(record_mask, with_kwargs=True)
            for layer in qeff.model.model.layers
        ]
        try:
            with torch.no_grad():
                qeff.model(
                    input_ids=torch.ones((1, 4), dtype=torch.int64),
                    target_hidden=torch.randn(1, 4, HIDDEN_SIZE),
                    position_ids_target=torch.arange(5, 9).unsqueeze(0),
                    position_ids=torch.arange(9, 13).unsqueeze(0),
                    past_key_values=QEffDynamicCache.from_legacy_cache(
                        _make_legacy_past_key_values(hf_model.config, 1, CTX_LEN)
                    ),
                    use_cache=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        for i, mask in enumerate(masks):
            causal = i == 0 if is_causal is None else is_causal
            for row, query in enumerate(range(9, 13)):
                for key in range(CTX_LEN):
                    visible = key <= 12 and (not causal or key <= query)
                    if i == 0:
                        visible = visible and query - key < 8 and (causal or key - query < 8)
                    assert mask[0, 0, row, key].item() == (not visible)


@pytest.mark.parametrize("start", [0, 2032, 2047, 2048, 2064])
@pytest.mark.parametrize("sliding_window,is_causal", [(None, False), (2048, True), (2048, False)])
def test_dflash_mask_crosses_sliding_window(start, sliding_window, is_causal):
    from QEfficient.transformers.models.dflash_draft.modeling_dflash_draft import _create_mask

    positions = torch.arange(start, start + 16).unsqueeze(0)
    mask = _create_mask(positions, 4096, sliding_window=sliding_window, is_causal=is_causal)
    keys = torch.arange(4096)
    for row, query in enumerate(positions[0].tolist()):
        visible = keys <= start + 15
        if is_causal:
            visible &= keys <= query
        if sliding_window:
            visible &= query - keys < sliding_window
            if not is_causal:
                visible &= keys - query < sliding_window
        torch.testing.assert_close(mask[0, 0, row], ~visible)


# ---------------------------------------------------------------------------
# Tests: target (TLM) side -- DFlashTLMTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestDFlashTLMTransform:
    """DFlashTLMTransform on the target (TLM) side -- multiple model families."""

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_dflash_tlm_projection_matches_model_precision(self, make_model, label, dtype):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states

        qeff_model = QEFFAutoModelForCausalLM(make_model().to(dtype=dtype), qaic_config={"target_layer_ids": [1]})
        inner = qeff_model.model.model
        target_hidden = torch.randn(1, 2, HIDDEN_SIZE, dtype=dtype)

        assert inner.fc.weight.dtype is dtype, f"[{label}] fc dtype does not match model dtype"
        assert inner.hidden_norm.weight.dtype is dtype, f"[{label}] hidden_norm dtype does not match model dtype"
        projected = compute_dflash_target_hidden_states([target_hidden], inner.fc, inner.hidden_norm)
        assert projected.dtype is dtype
        assert torch.isfinite(projected).all()

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    def test_dflash_tlm_transform_attaches_fc_and_hidden_norm(self, make_model, label):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_model()
        target_layer_ids = [1]
        qeff_model = QEFFAutoModelForCausalLM(hf_model, qaic_config={"target_layer_ids": target_layer_ids})

        assert qeff_model.dflash_tlm is True, f"[{label}] dflash_tlm flag not set"
        inner = qeff_model.model.model
        assert isinstance(inner.fc, nn.Linear), f"[{label}] fc not attached"
        assert inner.fc.in_features == HIDDEN_SIZE * len(target_layer_ids)
        assert inner.fc.out_features == HIDDEN_SIZE
        assert hasattr(inner, "hidden_norm"), f"[{label}] hidden_norm not attached"
        assert inner.target_layer_ids == target_layer_ids

    @pytest.mark.parametrize("make_model,label", _TLM_MODEL_FACTORIES, ids=_TLM_MODEL_IDS)
    def test_dflash_tlm_forward_produces_target_hidden_states(self, make_model, label):
        from QEfficient import QEFFAutoModelForCausalLM

        hf_model = make_model()
        target_layer_ids = [1]
        qeff_model = QEFFAutoModelForCausalLM(hf_model, qaic_config={"target_layer_ids": target_layer_ids})

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
        qeff_inputs = {
            "input_ids": input_ids,
            "position_ids": torch.arange(6).unsqueeze(0),
            "past_key_values": _make_legacy_past_key_values(hf_model.config, batch=1, ctx_len=CTX_LEN),
        }
        with torch.no_grad():
            outputs = qeff_model.model(**qeff_inputs)

        assert outputs.hidden_states is not None, f"[{label}] no target hidden_states returned"
        assert outputs.hidden_states.shape == (1, 6, HIDDEN_SIZE), f"[{label}] unexpected target hidden_states shape"


# ---------------------------------------------------------------------------
# Tests: compute_dflash_target_hidden_states (pure function)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestComputeDFlashTargetHiddenStates:
    """Pure-function correctness of compute_dflash_target_hidden_states."""

    def test_matches_manual_reference(self):
        from QEfficient.transformers.spd.dflash import compute_dflash_target_hidden_states

        n_layers, hidden = 3, HIDDEN_SIZE
        target_hidden_list = [torch.randn(2, 5, hidden) for _ in range(n_layers)]
        fc = nn.Linear(n_layers * hidden, hidden, bias=False)
        hidden_norm = nn.RMSNorm(hidden)

        result = compute_dflash_target_hidden_states(target_hidden_list, fc, hidden_norm)

        expected = hidden_norm(fc(torch.cat(target_hidden_list, dim=-1)))
        assert result.shape == (2, 5, hidden)
        assert torch.equal(result, expected)
