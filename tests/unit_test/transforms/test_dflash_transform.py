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

All tests run on CPU only, using tiny in-memory models and synthetic local safetensors checkpoints --
no downloaded checkpoints or Hugging Face Hub access.
"""

import copy
import io

import numpy as np
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
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("padding", [0, 2])
def test_qwen3_5_rope_helper_matches_hf_and_qeff(moe, with_image, padding):
    from types import SimpleNamespace

    from QEfficient.generation import dflash_generation
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import (
        QEffQwen3_5ForConditionalGeneration,
        QEffQwen3_5Model,
    )
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeForConditionalGeneration,
        QEffQwen3_5MoeModel,
    )

    with torch.device("meta"):
        hf_model = make_tiny_qwen3_5_vlm(moe=moe)
    qeff_model_cls = QEffQwen3_5MoeModel if moe else QEffQwen3_5Model
    qeff_generation_cls = QEffQwen3_5MoeForConditionalGeneration if moe else QEffQwen3_5ForConditionalGeneration
    assert qeff_model_cls.get_rope_index is type(hf_model.model).get_rope_index
    assert qeff_model_cls.get_vision_position_ids is type(hf_model.model).get_vision_position_ids

    tokens = [10, 5, 3, 3, 3, 3, 6, 11, 12] if with_image else [10, 11, 12, 13]
    input_ids = torch.tensor([tokens + [0] * padding], dtype=torch.int64)
    attention_mask = torch.tensor([[1] * len(tokens) + [0] * padding], dtype=torch.int64)
    mm_token_type_ids = (input_ids == hf_model.config.image_token_id).to(torch.int64)
    image_grid_thw = torch.tensor([[1, 4, 4]]) if with_image else None
    expected_mrope, expected_deltas = hf_model.model.get_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    positions, deltas = dflash_generation.compute_position_ids_qwen3_vl(
        input_ids, attention_mask, image_grid_thw, hf_model.config
    )
    torch.testing.assert_close(positions[1:], expected_mrope)
    torch.testing.assert_close(deltas, expected_deltas)
    prepared = qeff_generation_cls.prepare_inputs_for_generation(
        SimpleNamespace(model=hf_model.model),
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
        },
        prefill_seq_len=input_ids.shape[1],
    )
    torch.testing.assert_close(positions, prepared["position_ids"])

    hf_model.model.rope_deltas = expected_deltas
    native_decode = hf_model._prepare_position_ids_for_generation(
        torch.zeros((1, 4), dtype=torch.int64),
        {"past_key_values": SimpleNamespace(get_seq_length=lambda: len(tokens))},
    )
    flat_positions = np.arange(len(tokens), len(tokens) + 4).reshape(1, -1)
    decode_positions = dflash_generation.build_decode_position_ids_qwen3_vl(flat_positions, int(deltas[0, 0]))
    np.testing.assert_array_equal(decode_positions[0], flat_positions)
    np.testing.assert_array_equal(decode_positions[1:], native_decode[:, :, -4:].expand(3, -1, -1).numpy())


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
    assert wrapper.qaic_config is qaic_config
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
    assert qeff.lang_model.hash_params["dflash_cache_protocol"].endswith("fused_replay_kv_commit_v2")
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
            before = copy.deepcopy(cache)
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
                for layer in range(len(cache)):
                    torch.testing.assert_close(output.past_key_values[layer], before[layer], rtol=0, atol=0)
            else:
                # An independent prefix-only call is the retained-state oracle,
                # not a second target call in the production decoding path.
                committed = model(
                    input_ids=torch.cat([replay_ids, ids], dim=1),
                    position_ids=torch.cat([replay_positions, torch.full_like(pos, -1)], dim=1),
                    past_key_values=copy.deepcopy(before),
                    cache_commit=torch.ones(1, dtype=torch.long),
                    use_cache=True,
                )
                for layer in range(len(cache)):
                    torch.testing.assert_close(
                        output.past_key_values[layer], committed.past_key_values[layer], rtol=2e-4, atol=2e-4
                    )
            for layer in (1, 3):
                for actual_state, old_state in zip(output.past_key_values[layer], before[layer]):
                    cache_positions = torch.arange(old_state.shape[2])
                    updated = (cache_positions[None, :] == replay_positions.reshape(-1, 1)).any(dim=0)
                    torch.testing.assert_close(actual_state[:, :, ~updated], old_state[:, :, ~updated], rtol=0, atol=0)
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


@pytest.mark.parametrize("moe", [False, True])
def test_qwen3_5_dflash_retained_kv_export_uses_commit_input(moe):
    import onnx

    from QEfficient import QEFFAutoModelForImageTextToText
    from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5DynamicCache
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeDynamicCache

    qeff = QEFFAutoModelForImageTextToText(
        make_tiny_qwen3_5_vlm(moe=moe), kv_offload=True, qaic_config={"target_layer_ids": [1, 3]}
    )
    model = qeff.lang_model.model.language_model
    cache_cls = QEffQwen3_5MoeDynamicCache if moe else QEffQwen3_5DynamicCache

    class AttentionExport(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = model.layers[1].self_attn

        def forward(self, hidden, past_key, past_value, positions, cache_commit):
            cache = cache_cls(model.config)
            cache.kv_layers[1].keys, cache.kv_layers[1].values = past_key, past_value
            cos = hidden.new_ones((1, hidden.shape[1], self.attention.head_dim))
            output, _ = self.attention(
                hidden_states=hidden,
                position_embeddings=(cos, torch.zeros_like(cos)),
                attention_mask=_create_causal_mask(positions[0], past_key.shape[2]),
                past_key_values=cache,
                position_ids=positions,
                dflash_chunk_size=16,
                dflash_commit_final=cache_commit,
            )
            return output, *cache[1]

    inputs = (
        torch.randn(1, 32, 32),
        torch.randn(1, 1, 64, 16),
        torch.randn(1, 1, 64, 16),
        torch.arange(32).reshape(1, 1, -1),
        torch.ones(1, dtype=torch.long),
    )
    buffer = io.BytesIO()
    torch.onnx.export(
        AttentionExport().eval(),
        inputs,
        buffer,
        opset_version=17,
        dynamo=False,
        input_names=["hidden", "past_key", "past_value", "positions", "cache_commit"],
        output_names=["hidden_out", "past_key_out", "past_value_out"],
    )
    graph = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    producers = {output: node for node in graph.graph.node for output in node.output}
    for output in ("past_key_out", "past_value_out"):
        assert producers[output].op_type == "Where"
        pending = [producers[output].input[0]]
        seen = set()
        while pending:
            name = pending.pop()
            if name not in seen and name in producers:
                pending.extend(producers[name].input)
            seen.add(name)
        assert "cache_commit" in seen


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


def test_dflash2_dlm_dispatch_accepts_subclasses(monkeypatch):
    from transformers import Qwen3Config

    from QEfficient import QEFFAutoModelForCausalLM
    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM
    from QEfficient.transformers.models.pytorch_transforms import (
        DFlash2DLMTransform,
        DFlashDLMTransform,
        DFlashTransform,
        SamplerTransform,
        SpDTransform,
    )

    class WrappedDFlash2ForCausalLM(DFlash2ForCausalLM):
        pass

    config = Qwen3Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=32,
        head_dim=16,
        layer_types=["sliding_attention"],
        sliding_window=16,
        dflash_config={
            "block_size": 4,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 63,
            "selector_rank": 8,
            "selector_top_k": 1,
        },
    )
    model = WrappedDFlash2ForCausalLM(config).eval()
    applied = []
    dflash2_apply = DFlash2DLMTransform.apply

    def record_dflash2(cls, transformed_model, qaic_config=None, **kwargs):
        applied.append(type(transformed_model))
        return dflash2_apply(transformed_model, qaic_config, **kwargs)

    def reject_legacy(*args, **kwargs):
        raise AssertionError("A DFlash-2 subclass must not use the DFlash-1 transform path")

    def passthrough(cls, transformed_model, qaic_config=None, **kwargs):
        return transformed_model, False

    monkeypatch.setattr(DFlash2DLMTransform, "apply", classmethod(record_dflash2))
    monkeypatch.setattr(DFlashTransform, "apply", classmethod(reject_legacy))
    monkeypatch.setattr(DFlashDLMTransform, "apply", classmethod(reject_legacy))
    monkeypatch.setattr(SpDTransform, "apply", classmethod(passthrough))
    monkeypatch.setattr(SamplerTransform, "apply", classmethod(passthrough))

    wrapped = QEFFAutoModelForCausalLM(model, qaic_config={"dflash_dlm": True})

    assert wrapped.model is model
    assert applied == [WrappedDFlash2ForCausalLM]


def test_dflash2_target_transform_sets_shared_dflash_chunk_metadata():
    from QEfficient.transformers.models.pytorch_transforms import DFlash2TLMTransform

    model = make_tiny_qwen3()
    transformed, applied = DFlash2TLMTransform.apply(
        model,
        {"target_layer_ids": [1], "dflash_block_size": 8},
    )

    assert applied is True
    assert transformed.model.target_layer_ids == [1]
    assert transformed.model.dflash_chunk_size == 8


def _dflash_export_hash(wrapper):
    from QEfficient.utils.export_utils import _generate_export_hash

    wrapper.transform(ctx_len=CTX_LEN, seq_len=4, qaic_config=None)
    assert "qaic_config" not in wrapper.hash_params
    return _generate_export_hash(wrapper, ({}, ["logits"], {}), {}, type(wrapper)._export)[0]


def test_dflash2_dlm_hash_covers_effective_transform_options(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3Config

    from QEfficient import QEFFAutoModelForCausalLM
    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=32,
        head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        dflash_config={
            "block_size": 4,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 63,
            "selector_rank": 8,
            "selector_top_k": 4,
        },
    )

    def wrap(**options):
        torch.manual_seed(23)
        model = DFlash2ForCausalLM(copy.deepcopy(config)).eval()
        return QEFFAutoModelForCausalLM(model, qaic_config={"dflash_dlm": True, **options})

    default = wrap()
    explicit_default = wrap(dflash_residual_scale="4", dflash_emit_selector_hidden=False)
    unscaled = wrap(dflash_residual_scale=1)
    host_selector = wrap(dflash_emit_selector_hidden=True)
    assert _dflash_export_hash(default) == _dflash_export_hash(explicit_default)
    assert len({_dflash_export_hash(wrapper) for wrapper in (default, unscaled, host_selector)}) == 3
    assert default.hash_params["dflash_architecture"] == "dflash2"
    assert default.hash_params["dflash_residual_scale"] == default.model.model.residual_scale == 4.0
    assert default.hash_params["dflash_emit_selector_hidden"] is default.model.emit_selector_hidden is False
    assert host_selector.hash_params["dflash_emit_selector_hidden"] is host_selector.model.emit_selector_hidden is True
    assert default.model.model.layers[1].input_layernorm.variance_epsilon != (
        unscaled.model.model.layers[1].input_layernorm.variance_epsilon
    )

    source_hashes = []
    for source_name, value in (("target_a", 1.0), ("target_b", 2.0)):
        source = tmp_path / source_name
        source.mkdir()
        save_file({"model.embed_tokens.weight": torch.full((64, 32), value)}, source / "model.safetensors")
        wrapper = wrap(dflash_tlm_repo=str(source))
        assert wrapper.hash_params["dflash_tlm_repo"] == str(source)
        torch.testing.assert_close(wrapper.model.model.embed_tokens.weight, torch.full((64, 32), value))
        source_hashes.append(_dflash_export_hash(wrapper))
    assert source_hashes[0] != source_hashes[1]

    greedy_config = copy.deepcopy(config)
    greedy_config.dflash_config["selector_top_k"] = 1
    greedy = QEFFAutoModelForCausalLM(
        DFlash2ForCausalLM(greedy_config).eval(),
        qaic_config={"dflash_dlm": True, "dflash_emit_selector_hidden": True},
    )
    assert greedy.hash_params["dflash_emit_selector_hidden"] is greedy.model.emit_selector_hidden is False


@pytest.mark.parametrize("target_kind", ["text", "qwen3_5", "qwen3_5_moe"])
def test_dflash2_tlm_hash_covers_checkpoint_and_projection_options(tmp_path, target_kind):
    from safetensors.torch import save_file
    from transformers import Qwen3Config

    from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
    from QEfficient.utils.constants import _DFLASH_TARGET_ABSMAX

    def make_model():
        torch.manual_seed(23)
        return make_tiny_qwen3() if target_kind == "text" else make_tiny_qwen3_5_vlm(target_kind == "qwen3_5_moe")

    model = make_model()
    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    sources = []
    for source_name, value in (("draft_a", 100.0), ("draft_b", 200.0)):
        source = tmp_path / source_name
        source.mkdir()
        Qwen3Config(hidden_size=hidden_size, dflash_config={"target_layer_ids": [0], "block_size": 8}).save_pretrained(
            source
        )
        save_file(
            {"fc.weight": torch.eye(hidden_size) * value, "hidden_norm.weight": torch.ones(hidden_size)},
            source / "model.safetensors",
        )
        sources.append(str(source))

    def wrap(checkpoint=sources[0], **options):
        model = make_model()
        qaic_config = {"dflash2_checkpoint": checkpoint, **options}
        if target_kind == "text":
            return QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)
        return QEFFAutoModelForImageTextToText(model, kv_offload=True, qaic_config=qaic_config).lang_model

    default = wrap()
    explicit_default = wrap(dflash_target_absmax=str(_DFLASH_TARGET_ABSMAX))
    smaller_bound = wrap(dflash_target_absmax=_DFLASH_TARGET_ABSMAX / 2)
    other_checkpoint = wrap(checkpoint=sources[1])
    zero_centered = wrap(dflash_hidden_norm="ZERO_CENTERED")
    assert _dflash_export_hash(default) == _dflash_export_hash(explicit_default)
    assert (
        len({_dflash_export_hash(wrapper) for wrapper in (default, smaller_bound, other_checkpoint, zero_centered)})
        == 4
    )
    assert default.hash_params["dflash2_checkpoint"] == sources[0]
    assert default.hash_params["dflash_target_absmax"] == _DFLASH_TARGET_ABSMAX
    assert zero_centered.hash_params["dflash_hidden_norm"] == "zero_centered"
    text_model = default.model.model
    bounded_text_model = smaller_bound.model.model
    if target_kind != "text":
        text_model = text_model.model.language_model
        bounded_text_model = bounded_text_model.model.language_model
    torch.testing.assert_close(bounded_text_model.fc.weight * 2, text_model.fc.weight)


@pytest.mark.parametrize("target_kind", ["text", "qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("override_layers", [False, True])
def test_dflash2_target_wrappers_resolve_checkpoint_metadata(tmp_path, target_kind, override_layers):
    from safetensors.torch import save_file
    from transformers import Qwen3Config

    from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText

    torch.manual_seed(23)
    model = make_tiny_qwen3() if target_kind == "text" else make_tiny_qwen3_5_vlm(target_kind == "qwen3_5_moe")
    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    config = Qwen3Config(hidden_size=hidden_size, dflash_config={"target_layer_ids": [0], "block_size": 8})
    config.save_pretrained(tmp_path)
    weights = {"fc.weight": torch.eye(hidden_size), "hidden_norm.weight": torch.ones(hidden_size)}
    save_file(weights, tmp_path / "model.safetensors")
    qaic_config = {"dflash2_checkpoint": str(tmp_path), "dflash2_target_layer_offset": 1}
    if override_layers:
        qaic_config["target_layer_ids"] = [1]

    if target_kind == "text":
        qeff = QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)
        text_model = qeff.model.model
        hash_params = qeff.hash_params
    else:
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=True, qaic_config=qaic_config)
        text_model = qeff.model.model.language_model
        hash_params = qeff.lang_model.hash_params

    assert qeff.dflash_tlm
    assert qaic_config["target_layer_ids"] == [1]
    assert qaic_config["dflash_block_size"] == 8
    assert text_model.target_layer_ids == [1]
    assert text_model.dflash_chunk_size == 8
    assert hash_params["target_layer_ids"] == (1,)
    if target_kind != "text":
        assert hash_params["dflash_chunk_size"] == 8
    torch.testing.assert_close(text_model.fc.weight, weights["fc.weight"])
    torch.testing.assert_close(text_model.hidden_norm.weight, weights["hidden_norm.weight"])


def test_dflash2_state_dict_remap_and_export_submodule():
    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import (
        DFlash2DecoderLayer,
        DFlash2ForCausalLM,
        remap_dflash2_state_dict,
    )

    remapped = remap_dflash2_state_dict(
        {
            "layers.0.input_layernorm.weight": torch.ones(4),
            "candidate_selector.predecessor_codebook": torch.ones(8, 2),
            "fc.weight": torch.ones(4, 4),
        }
    )

    assert "model.layers.0.input_layernorm.weight" in remapped
    assert "model.candidate_selector.predecessor_codebook.weight" in remapped
    assert "fc.weight" not in remapped
    assert DFlash2ForCausalLM.__new__(DFlash2ForCausalLM).get_submodules_for_export() == {DFlash2DecoderLayer}


def test_dflash2_dlm_transform_scales_fp16_residual_stream():
    from transformers import Qwen3Config

    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM
    from QEfficient.transformers.models.pytorch_transforms import DFlash2DLMTransform

    torch.manual_seed(23)
    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        layer_types=["sliding_attention", "sliding_attention"],
        dflash_config={
            "block_size": 4,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB_SIZE - 1,
            "selector_rank": 8,
            "selector_top_k": 1,
        },
    )
    model = DFlash2ForCausalLM(config).eval()
    for layer in model.model.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            nn.init.normal_(conv.base_kernel, std=config.initializer_range)
    inputs = {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, 4)),
        "target_hidden": torch.randn(1, 4, HIDDEN_SIZE),
        "position_ids_target": torch.arange(4).unsqueeze(0),
        "position_ids": torch.arange(4, 8).unsqueeze(0),
        "attention_mask": torch.zeros(1, 8),
        "use_cache": True,
    }
    with torch.no_grad():
        reference = model(past_key_values=_make_legacy_past_key_values(config, 1, 8), **inputs)
    assert torch.isfinite(reference.logits).all()
    epsilon = model.model.norm.variance_epsilon
    finish_weights = [
        layer.attention_conv.kernel_projection.weight[layer.attention_conv.kernel_size * layer.attention_conv.groups :]
        .detach()
        .clone()
        for layer in model.model.layers
    ]

    transformed, applied = DFlash2DLMTransform.apply(
        model,
        {"dflash_dlm": True, "dflash_residual_scale": 4.0},
    )

    assert applied is True
    assert transformed.model.residual_scale == 4.0
    assert transformed.model.layers[0].scale_input_residual is True
    assert transformed.model.layers[1].scale_input_residual is False
    assert transformed.model.layers[0].input_layernorm.variance_epsilon == epsilon
    assert transformed.model.layers[0].post_attention_layernorm.variance_epsilon == epsilon / 16
    assert transformed.model.layers[1].input_layernorm.variance_epsilon == epsilon / 16
    assert transformed.model.norm.variance_epsilon == epsilon / 16
    for layer, original_finish_weight in zip(transformed.model.layers, finish_weights):
        finish_offset = layer.attention_conv.kernel_size * layer.attention_conv.groups
        assert torch.equal(
            layer.attention_conv.kernel_projection.weight[finish_offset:],
            original_finish_weight / 4,
        )

    with torch.no_grad():
        actual = transformed(past_key_values=_make_legacy_past_key_values(config, 1, 8), **inputs)
    assert actual.logits.shape == (1, 4, VOCAB_SIZE)
    assert torch.isfinite(actual.logits).all()
    torch.testing.assert_close(actual.logits, reference.logits, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_dflash2_legacy_export_preserves_dynamic_sliding_window(causal):
    onnxruntime = pytest.importorskip("onnxruntime")

    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import _create_mask

    class MaskExport(nn.Module):
        def forward(self, positions, cache):
            return _create_mask(positions, cache.shape[2], causal=causal, sliding_window=2048)

    model = MaskExport().eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (torch.arange(8, 16).reshape(1, 8), torch.zeros(1, 1, 16, 1)),
        buffer,
        input_names=["positions", "cache"],
        output_names=["mask"],
        dynamic_axes={
            "positions": {1: "seq_len"},
            "cache": {2: "ctx_len"},
            "mask": {2: "seq_len", 3: "ctx_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    positions = torch.arange(2050, 2058).reshape(1, 8)
    cache = torch.zeros(1, 1, 4096, 1)
    expected = model(positions, cache).numpy()
    assert expected[0, 0, 0, 0]
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        buffer.getvalue(), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"positions": positions.numpy(), "cache": cache.numpy()})[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("comp_ctx_len", [12, 16])
def test_dflash2_attention_gathers_comp_ctx_lengths(comp_ctx_len):
    from transformers import Qwen3Config

    from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
    from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2Attention, _create_mask

    torch.manual_seed(23)
    config = Qwen3Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=32,
        head_dim=16,
        use_sliding_window=True,
        sliding_window=32,
        layer_types=["sliding_attention"],
    )
    attention = DFlash2Attention(config, layer_idx=0).eval()
    cache = QEffSlidingWindowCache(config, batch_size=1, max_cache_len=32, sliding_window_len=32)
    cache.key_cache.append(torch.zeros(1, 1, 32, 16))
    cache.value_cache.append(torch.zeros(1, 1, 32, 16))
    positions = torch.arange(8, 12).reshape(1, 4)
    cosine = torch.ones(1, 8, 16)
    with torch.no_grad():
        output, weights = attention(
            hidden_states=torch.randn(1, 4, 32),
            target_hidden=torch.randn(1, 4, 32),
            position_ids_target=torch.arange(4, 8).reshape(1, 4),
            position_ids=positions,
            attention_mask=_create_mask(positions, 32, causal=False, sliding_window=32),
            past_key_value=cache,
            comp_ctx_lengths=torch.zeros(comp_ctx_len, dtype=torch.int64),
            position_embeddings=(cosine, torch.zeros_like(cosine)),
        )
    assert output.shape == (1, 4, config.hidden_size)
    assert weights.shape == (1, config.num_attention_heads, 4, comp_ctx_len)
    assert cache.key_cache[0].shape[2] == 32
    assert cache.value_cache[0].shape[2] == 32
    assert torch.isfinite(output).all()


@pytest.mark.transforms
class TestDFlash2CheckpointLoading:
    @pytest.fixture(autouse=True)
    def forbid_pickle_loading(self, monkeypatch):
        def fail_pickle_load(*args, **kwargs):
            pytest.fail("DFlash-2 checkpoints must not be loaded with pickle")

        monkeypatch.setattr(torch, "load", fail_pickle_load)

    @pytest.fixture
    def tiny_dflash2_checkpoint(self, tmp_path):
        from safetensors.torch import save_file
        from transformers import Qwen3Config

        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

        config = Qwen3Config(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=32,
            intermediate_size=64,
            vocab_size=64,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=32,
            layer_types=["sliding_attention"],
            architectures=["DFlash2DraftModel"],
            dtype=torch.bfloat16,
            dflash_config={
                "block_size": 4,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 63,
                "selector_rank": 4,
                "selector_top_k": 1,
            },
        )
        model = DFlash2ForCausalLM(config)
        for layer in model.model.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                nn.init.normal_(conv.base_kernel, std=config.initializer_range)
        weights = {}
        for key, tensor in model.state_dict().items():
            if key in {"model.embed_tokens.weight", "lm_head.weight"}:
                continue
            key = key.removeprefix("model.")
            if key in {
                "candidate_selector.predecessor_codebook.weight",
                "candidate_selector.successor_codebook.weight",
            }:
                key = key.removesuffix(".weight")
            weights[key] = tensor.to(torch.bfloat16).contiguous()
        checkpoint = tmp_path / "draft"
        checkpoint.mkdir()
        config.save_pretrained(checkpoint)
        save_file(weights, checkpoint / "model.safetensors")
        return checkpoint, config, weights

    @pytest.mark.parametrize("loader_kind", ["native", "auto"])
    @pytest.mark.parametrize("dtype,expected_dtype", [("float16", torch.float16), ("auto", torch.bfloat16)])
    def test_loads_hf_string_dtype(self, tiny_dflash2_checkpoint, loader_kind, dtype, expected_dtype):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

        checkpoint, _, weights = tiny_dflash2_checkpoint
        if loader_kind == "auto":
            wrapper = QEFFAutoModelForCausalLM.from_pretrained(
                checkpoint, dtype=dtype, qaic_config={"dflash_dlm": True, "dflash_residual_scale": 1.0}
            )
            model = wrapper.model
        else:
            model = DFlash2ForCausalLM.from_pretrained(checkpoint, dtype=dtype)

        assert model.model.norm.weight.dtype == expected_dtype
        assert model.config.dtype == expected_dtype
        torch.testing.assert_close(model.model.norm.weight, weights["norm.weight"].to(expected_dtype))

    def test_auto_dtype_falls_back_to_checkpoint(self, tiny_dflash2_checkpoint):
        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

        checkpoint, config, weights = tiny_dflash2_checkpoint
        config.dtype = None

        model = DFlash2ForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype="auto")

        assert model.model.norm.weight.dtype == torch.bfloat16
        assert model.config.dtype == torch.bfloat16
        torch.testing.assert_close(model.model.norm.weight, weights["norm.weight"])

    def test_auto_loader_uses_supplied_config(self, tiny_dflash2_checkpoint):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

        checkpoint, config, weights = tiny_dflash2_checkpoint
        (checkpoint / "config.json").unlink()

        wrapper = QEFFAutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            dtype=torch.float32,
            qaic_config={"dflash_dlm": True, "dflash_residual_scale": 1.0},
        )

        assert isinstance(wrapper.model, DFlash2ForCausalLM)
        assert wrapper.model.config is config
        torch.testing.assert_close(
            wrapper.model.model.layers[0].attention_conv.base_kernel,
            weights["layers.0.attention_conv.base_kernel"].float(),
        )

    @pytest.mark.parametrize("architectures", [None, [], ["UnknownDraftModel"]])
    def test_rejects_dflash2_config_without_supported_architecture(
        self, tiny_dflash2_checkpoint, monkeypatch, architectures
    ):
        from unittest.mock import Mock

        from QEfficient import QEFFAutoModelForCausalLM

        checkpoint, config, _ = tiny_dflash2_checkpoint
        config.architectures = architectures
        fallback = Mock(side_effect=AssertionError("DFlash-2 must not fall back to the HF loader"))
        monkeypatch.setattr(QEFFAutoModelForCausalLM._hf_auto_class, "from_pretrained", fallback)

        with pytest.raises(ValueError, match="DFlash2DraftModel"):
            QEFFAutoModelForCausalLM.from_pretrained(
                checkpoint,
                config=config,
                qaic_config={"dflash_dlm": True, "dflash_residual_scale": 1.0},
            )
        fallback.assert_not_called()

    def test_auto_loader_preserves_configuration_errors(self, monkeypatch):
        from unittest.mock import Mock

        from QEfficient import QEFFAutoModelForCausalLM

        config_loader = Mock(side_effect=OSError("checkpoint configuration could not be read"))
        fallback = Mock(side_effect=AssertionError("Configuration errors must not trigger an HF fallback"))
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", config_loader)
        monkeypatch.setattr(QEFFAutoModelForCausalLM._hf_auto_class, "from_pretrained", fallback)

        with pytest.raises(OSError, match="checkpoint configuration could not be read"):
            QEFFAutoModelForCausalLM.from_pretrained("test-org/unreadable-checkpoint", local_files_only=True)
        fallback.assert_not_called()

    @pytest.mark.parametrize("loader_kind", ["native", "auto"])
    def test_loads_local_subfolder(self, tiny_dflash2_checkpoint, loader_kind):
        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import DFlash2ForCausalLM

        checkpoint, _, weights = tiny_dflash2_checkpoint
        kwargs = {"subfolder": checkpoint.name, "local_files_only": True, "dtype": torch.float32}
        if loader_kind == "auto":
            wrapper = QEFFAutoModelForCausalLM.from_pretrained(
                checkpoint.parent,
                qaic_config={"dflash_dlm": True, "dflash_residual_scale": 1.0},
                **kwargs,
            )
            model = wrapper.model
        else:
            model = DFlash2ForCausalLM.from_pretrained(checkpoint.parent, **kwargs)

        assert isinstance(model, DFlash2ForCausalLM)
        torch.testing.assert_close(model.model.norm.weight, weights["norm.weight"].float())

    @pytest.mark.parametrize("local_files_only,force_download", [(True, False), (False, True)])
    def test_auto_loader_forwards_hub_options(
        self, tiny_dflash2_checkpoint, monkeypatch, local_files_only, force_download
    ):
        from unittest.mock import Mock, call

        from QEfficient import QEFFAutoModelForCausalLM

        checkpoint, config, weights = tiny_dflash2_checkpoint
        config_loader = Mock(return_value=config)
        snapshot_download = Mock(return_value=str(checkpoint.parent))
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", config_loader)
        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
        hub_kwargs = {
            "cache_dir": str(checkpoint.parent / "cache"),
            "revision": "test-revision",
            "token": "test-token",
            "subfolder": checkpoint.name,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        wrapper = QEFFAutoModelForCausalLM.from_pretrained(
            "test-org/dflash2-draft",
            dtype="auto",
            qaic_config={"dflash_dlm": True, "dflash_residual_scale": 1.0},
            **hub_kwargs,
        )

        expected_config_call = call("test-org/dflash2-draft", trust_remote_code=False, **hub_kwargs)
        assert config_loader.call_args_list == [expected_config_call, expected_config_call]
        download_kwargs = {key: value for key, value in hub_kwargs.items() if key != "subfolder"}
        snapshot_download.assert_called_once_with(
            "test-org/dflash2-draft", allow_patterns=[f"{checkpoint.name}/*.safetensors"], **download_kwargs
        )
        torch.testing.assert_close(wrapper.model.model.norm.weight, weights["norm.weight"])

    @pytest.mark.parametrize("shard_count", [1, 2])
    def test_loads_safetensors_shards(self, tmp_path, shard_count):
        from safetensors.torch import save_file

        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import _read_checkpoint_tensors

        expected = {}
        for shard_index in range(shard_count):
            tensors = {f"layers.{shard_index}.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            save_file(tensors, tmp_path / f"model-{shard_index}.safetensors")
            expected.update(tensors)
        (tmp_path / "pytorch_model.bin").write_bytes(b"not a safetensors checkpoint")

        actual = _read_checkpoint_tensors(tmp_path)

        assert actual.keys() == expected.keys()
        for key, tensor in expected.items():
            torch.testing.assert_close(actual[key], tensor)
            assert actual[key].device.type == "cpu"

    def test_downloads_only_safetensors(self, tmp_path, monkeypatch):
        from unittest.mock import Mock

        from safetensors.torch import save_file

        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import _read_checkpoint_tensors

        tensors = {"norm.weight": torch.ones(4)}
        save_file(tensors, tmp_path / "model.safetensors")
        snapshot_download = Mock(return_value=str(tmp_path))
        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
        hub_kwargs = {"revision": "test-revision", "cache_dir": str(tmp_path / "cache"), "token": "test-token"}

        actual = _read_checkpoint_tensors("test-org/dflash2-draft", **hub_kwargs)

        snapshot_download.assert_called_once_with(
            "test-org/dflash2-draft", allow_patterns=["*.safetensors"], **hub_kwargs
        )
        torch.testing.assert_close(actual["norm.weight"], tensors["norm.weight"])

    @pytest.mark.parametrize("checkpoint_format", ["missing", "bin", "empty_safetensors"])
    def test_rejects_checkpoints_without_safetensors_weights(self, tmp_path, checkpoint_format):
        from safetensors.torch import save_file

        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import _read_checkpoint_tensors

        if checkpoint_format == "bin":
            (tmp_path / "pytorch_model.bin").write_bytes(b"not a safetensors checkpoint")
        elif checkpoint_format == "empty_safetensors":
            save_file({}, tmp_path / "model.safetensors")

        with pytest.raises(FileNotFoundError, match="DFlash-2 requires safetensors checkpoints"):
            _read_checkpoint_tensors(tmp_path)

    def test_rejects_invalid_safetensors_without_pickle_fallback(self, tmp_path):
        from safetensors import SafetensorError

        from QEfficient.transformers.models.dflash2_draft.modeling_dflash2_draft import _read_checkpoint_tensors

        (tmp_path / "model.safetensors").write_bytes(b"not a safetensors checkpoint")
        (tmp_path / "pytorch_model.bin").write_bytes(b"not a safetensors checkpoint")

        with pytest.raises(SafetensorError):
            _read_checkpoint_tensors(tmp_path)


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


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "qwen3_5", "qwen3_5_moe"])
def test_dflash_tlm_hash_includes_target_layer_selection(model_type, monkeypatch):
    from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
    from QEfficient.utils.hash_utils import create_export_hash

    monkeypatch.setattr("QEfficient.transformers.models.pytorch_transforms.load_checkpoint_weights", lambda *args: {})

    def wrap(target_layer_ids=None):
        qaic_config = None
        if target_layer_ids is not None:
            qaic_config = {"target_layer_ids": target_layer_ids, "dflash_dlm_repo": "test/dflash-draft"}
        torch.manual_seed(7)
        if model_type in {"qwen3", "llama"}:
            model = make_tiny_qwen3() if model_type == "qwen3" else make_tiny_llama()
            return QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)
        model = make_tiny_qwen3_5_vlm(moe=model_type == "qwen3_5_moe")
        return QEFFAutoModelForImageTextToText(model, kv_offload=True, qaic_config=qaic_config).lang_model

    assert "target_layer_ids" not in wrap().hash_params
    hashes = []
    for selection in [(0,), (1,), (0, 1), (1, 0)]:
        equivalent_hashes = []
        for target_layer_ids in [list(selection), selection]:
            params = wrap(target_layer_ids).hash_params
            assert params["target_layer_ids"] == selection
            assert isinstance(params["target_layer_ids"], tuple)
            export_hash, _ = create_export_hash(model_params=copy.deepcopy(params))
            equivalent_hashes.append(export_hash)
        assert equivalent_hashes[0] == equivalent_hashes[1]
        hashes.append(equivalent_hashes[0])
    assert len(set(hashes)) == len(hashes)


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
