# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""Regression tests for continuous-batched DFlash VLM target execution."""

import copy

import pytest
import torch

from .test_dflash_transform import make_tiny_qwen3_5_vlm


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("ccl", [False, True])
@pytest.mark.parametrize("dflash", [False, True])
def test_qwen35_batched_verification_specializations(moe, batch_size, ccl, dflash):
    from QEfficient import QEFFAutoModelForImageTextToText

    qeff = QEFFAutoModelForImageTextToText(
        make_tiny_qwen3_5_vlm(moe),
        kv_offload=True,
        continuous_batching=True,
        qaic_config={"target_layer_ids": [1, 3]} if dflash else {},
    )
    specs, _ = qeff.model.get_specializations(
        batch_size=1,
        full_batch_size=batch_size,
        kv_cache_batch_size=batch_size,
        continuous_batching=True,
        kv_offload=True,
        prefill_seq_len=128,
        ctx_len=512,
        comp_ctx_lengths_prefill=[256, 512] if ccl else None,
        comp_ctx_lengths_decode=[256, 512] if ccl else None,
    )
    for spec in specs["lang"]:
        decode = spec["seq_len"] == (32 if dflash else 1)
        assert spec["batch_size"] == (batch_size if decode else 1)
        assert spec["full_batch_size"] == batch_size
        if dflash:
            assert spec["num_logits_to_keep"] == (16 if decode else 1)
        else:
            assert "num_logits_to_keep" not in spec
        if decode:
            assert "full_batch_exec_size" not in spec


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("idle_slot", [1, -1])
@torch.no_grad()
def test_qwen35_batched_state_matches_independent_requests(moe, with_image, idle_slot):
    from QEfficient import QEFFAutoModelForImageTextToText

    torch.manual_seed(31)
    hf = make_tiny_qwen3_5_vlm(moe)
    hf.config.text_config.max_position_embeddings = 256
    qaic_config = {"target_layer_ids": [1, 3], "speculative_model_type": "target"}
    qeff = QEFFAutoModelForImageTextToText(
        hf,
        kv_offload=True,
        continuous_batching=True,
        qaic_config=qaic_config,
    )
    if moe:
        qeff.transform(
            ctx_len=256,
            seq_len=32,
            bs=1,
            num_devices=1,
            aic_num_cores=4,
            qaic_config=qaic_config,
            prefill_only=True,
            prefill_seq_len=32,
        )
    wrapper = qeff.lang_model.model
    dummy = qeff.model.get_dummy_inputs(
        kv_offload=True,
        continuous_batching=True,
        prefill_seq_len=32,
        ctx_len=256,
    )["lang"]
    zero = tuple(
        tuple(torch.zeros((4, *s.shape[1:]), dtype=s.dtype) for s in layer) for layer in dummy["past_key_values"]
    )
    state = copy.deepcopy(zero)
    refs = {slot: tuple(tuple(s[:1].clone() for s in layer) for layer in zero) for slot in (2, 0)}
    images = {slot: torch.randn(1, 2, 32) for slot in refs}
    lengths = {2: 5, 0: 7}
    prefixes = {slot: (torch.zeros(1, 16, dtype=torch.long), torch.full((1, 16), -1)) for slot in refs}

    def call(ids, positions, cache, image, commit, slots=None):
        return wrapper(
            input_ids=ids,
            position_ids=positions.unsqueeze(0).expand(4, -1, -1),
            past_key_values=copy.deepcopy(cache),
            vision_embeds=image,
            image_idx=torch.zeros((1, 1), dtype=torch.long),
            cache_commit=torch.tensor([commit]),
            num_logits_to_keep=torch.zeros((1 if commit else 16, 1), dtype=torch.long),
            batch_index=None if slots is None else torch.tensor(slots).reshape(-1, 1),
        )

    def compare_rows(actual, expected, row=0, valid_hidden=slice(-16, None)):
        torch.testing.assert_close(actual[0][row : row + 1], expected[0], atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(actual[0][row : row + 1].argmax(-1), expected[0].argmax(-1), atol=0, rtol=0)
        torch.testing.assert_close(
            actual[3][row : row + 1, valid_hidden], expected[3][:, valid_hidden], atol=3e-4, rtol=3e-4
        )

    def unchanged(actual, before, slots):
        for layer, old_layer in zip(actual, before):
            for tensor, old in zip(layer, old_layer):
                torch.testing.assert_close(tensor[slots], old[slots], atol=0, rtol=0)

    # Prefill one request at a time into nonconsecutive cache slots.
    for slot, length in lengths.items():
        ids = torch.randint(10, 500, (1, 32))
        if with_image:
            ids[:, :2] = hf.config.image_token_id
        pos = torch.arange(32).reshape(1, -1)
        pos[:, length:] = -1
        actual = call(ids, pos, state, images[slot], 1, [slot])
        expected = call(ids, pos, refs[slot], images[slot], 1)
        compare_rows(actual, expected, valid_hidden=slice(0, length))
        unchanged(actual[-1], state, [s for s in range(4) if s != slot])
        state, refs[slot] = actual[-1], expected[-1]

    # Reordered active rows have different accepted-prefix lengths.
    for order, accepted in [([2, 0], [2, 5]), ([0, 2], [16, 1]), ([2, 0], [3, 7])]:
        new_ids = torch.randint(10, 500, (2, 16))
        new_pos = torch.stack([torch.arange(lengths[s], lengths[s] + 16) for s in order])
        ids = torch.cat([torch.cat([prefixes[s][0] for s in order]), new_ids], dim=1)
        pos = torch.cat([torch.cat([prefixes[s][1] for s in order]), new_pos], dim=1)
        actual = call(ids, pos, state, images[order[0]], 0, order)
        unchanged(actual[-1], state, [1, 3])
        if not (pos[:, :16] >= 0).any():
            unchanged(actual[-1], state, [0, 1, 2, 3])
            repeated = call(ids, pos, actual[-1], images[order[0]], 0, order)
            unchanged(repeated[-1], state, [0, 1, 2, 3])
            torch.testing.assert_close(repeated[0], actual[0], atol=0, rtol=0)
        for row, slot in enumerate(order):
            expected = call(ids[row : row + 1], pos[row : row + 1], refs[slot], images[slot], 0)
            compare_rows(actual, expected, row)
            for layer, ref_layer in zip(actual[-1], expected[-1]):
                for tensor, ref in zip(layer, ref_layer):
                    torch.testing.assert_close(tensor[slot : slot + 1], ref, atol=3e-4, rtol=3e-4)
            refs[slot] = expected[-1]
            prefix_pos = new_pos[row : row + 1].clone()
            prefix_pos[:, accepted[row] :] = -1
            prefixes[slot] = new_ids[row : row + 1], prefix_pos
            lengths[slot] += accepted[row]
        state = actual[-1]

    # An all-padding row must leave its cache untouched.
    ids = torch.cat([prefixes[2][0], torch.zeros(1, 16, dtype=torch.long)], dim=1).repeat(2, 1)
    pos = torch.cat([prefixes[2][1], torch.full((1, 16), -1)], dim=1).repeat(2, 1)
    pos[1] = -1
    actual = call(ids, pos, state, images[2], 0, [2, idle_slot])
    unchanged(actual[-1], state, [0, 1, 3])
    state = actual[-1]

    # Reusing a slot resets that request's hybrid state, not other requests.
    ids = torch.randint(10, 500, (1, 32))
    if with_image:
        ids[:, :2] = hf.config.image_token_id
    pos = torch.arange(32).reshape(1, -1)
    pos[:, 6:] = -1
    actual = call(ids, pos, state, images[0], 1, [0])
    expected = call(ids, pos, tuple(tuple(s[:1].clone() for s in layer) for layer in zero), images[0], 1)
    compare_rows(actual, expected, valid_hidden=slice(0, 6))
    unchanged(actual[-1], state, [1, 2, 3])
