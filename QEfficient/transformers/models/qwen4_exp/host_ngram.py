# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""Host-side PLE n-gram hashing and sharded embedding lookup for Qwen4-Exp.

The checkpoint's PLE table is intentionally not a model parameter in the QEff
decode graph.  This module reads just the requested rows from its safetensors
shards and returns the resulting activation to the caller.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1
_PRIME_1 = 10007
_SHARD_KEY = re.compile(r"^(?P<prefix>.+)\.shard_(?P<index>\d+)\.weight$")


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> torch.Tensor:
    """Match Transformers' private Qwen4-Exp multiplier derivation exactly."""
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class HostNGramHistory:
    """CPU-only previous-token state used before each host PLE lookup."""

    def __init__(self, ngram_size: int, eos_token_id: int):
        self.context_len = ngram_size - 1
        self.eos_token_id = eos_token_id
        self.token_ids: torch.Tensor | None = None

    def previous(self, batch_size: int) -> torch.Tensor:
        if self.token_ids is None:
            return torch.full((batch_size, self.context_len), self.eos_token_id, dtype=torch.long)
        if self.token_ids.shape[0] != batch_size:
            raise ValueError("Host n-gram history batch size cannot change during decode")
        return self.token_ids

    def update(self, input_ids: torch.Tensor) -> None:
        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long)
        history = torch.cat((self.previous(input_ids.shape[0]), input_ids), dim=-1)
        self.token_ids = history[:, -self.context_len :].clone()


class ShardedNGramLookup:
    """Reference PLE lookup which never materializes a full n-gram table."""

    def __init__(self, config, checkpoint_dir: str | Path, ple_layer_index: int = 0):
        self.config = getattr(config, "text_config", config)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.ple_layer_index = ple_layer_index
        self.ngram_size = self.config.ngram_size
        self.heads_per_ngram = self.config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = self.config.ple_embed_dim // self.ngram_heads
        self.eos_token_id = (
            self.config.eos_token_id[0] if isinstance(self.config.eos_token_id, list) else self.config.eos_token_id
        )
        self.layer_multipliers = _build_layer_multipliers(
            self.config.vocab_size, self.ngram_size, ple_layer_index, self.config.seed
        )
        sizes, offsets, total = [], [], 0
        for head_idx in range(self.ngram_heads):
            size = _find_nth_prime_after(
                self.config.ngram_vocab_size_base - 1, ple_layer_index * self.ngram_heads + head_idx + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
        self.head_offsets = torch.tensor(offsets, dtype=torch.long)
        self.padded_vocab_size = math.ceil(total / self.config.make_ngram_vocab_size_divisible_by) * (
            self.config.make_ngram_vocab_size_divisible_by
        )
        self._shards = self._discover_shards()
        self._shard_offsets, self.dtype = self._read_shard_offsets()

    def _discover_shards(self) -> list[tuple[str, str]]:
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        candidates = []
        for key, filename in weight_map.items():
            match = _SHARD_KEY.match(key)
            if match and ".ple_embedding.ngram_embedding" in match.group("prefix"):
                candidates.append((int(match.group("index")), key, filename))
        if not candidates:
            raise KeyError("No sharded Qwen4-Exp PLE embedding weights found in checkpoint index")
        # Each PLE module has its own n-gram table.  Index by the configured PLE order.
        prefixes = sorted({key.rsplit(".shard_", 1)[0] for _, key, _ in candidates})
        if self.ple_layer_index >= len(prefixes):
            raise IndexError(f"PLE table {self.ple_layer_index} is absent from checkpoint")
        prefix = prefixes[self.ple_layer_index]
        return [(key, filename) for _, key, filename in sorted(candidates) if key.startswith(f"{prefix}.shard_")]

    def _read_shard_offsets(self) -> tuple[list[int], torch.dtype]:
        offsets, total = [], 0
        dtype = None
        for key, filename in self._shards:
            with safe_open(str(self.checkpoint_dir / filename), framework="pt", device="cpu") as handle:
                offsets.append(total)
                tensor_slice = handle.get_slice(key)
                total += tensor_slice.get_shape()[0]
                shard_dtype = {
                    "BF16": torch.bfloat16,
                    "F16": torch.float16,
                    "F32": torch.float32,
                    "F64": torch.float64,
                }.get(tensor_slice.get_dtype())
                if shard_dtype is None:
                    raise TypeError(f"Unsupported PLE shard dtype {tensor_slice.get_dtype()}")
                if dtype is not None and dtype != shard_dtype:
                    raise TypeError("All PLE n-gram shards must have the same dtype")
                dtype = shard_dtype
        if total < self.padded_vocab_size:
            raise ValueError(f"PLE shards contain {total} rows, expected at least {self.padded_vocab_size}")
        return offsets, dtype

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat((eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]), dim=1)
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        shifted = token_ids.gather(1, source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1))
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def ngram_ids(self, input_ids: torch.Tensor, history: HostNGramHistory) -> torch.Tensor:
        """Return global PLE table IDs and leave history unchanged."""
        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long)
        token_history = torch.cat((history.previous(input_ids.shape[0]), input_ids), dim=-1)
        shifted = [self._shift_right_ignore_eos(token_history, offset) for offset in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            ids = torch.remainder(mixed.unsqueeze(-1), self.head_vocab_sizes[start : start + self.heads_per_ngram])
            blocks.append(ids + self.head_offsets[start : start + self.heads_per_ngram])
        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]

    def _lookup_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = row_ids.reshape(-1)
        result = torch.empty((flat_ids.numel(), self.head_dim), dtype=self.dtype)
        shard_ends = self._shard_offsets[1:] + [self.padded_vocab_size]
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for result_index, row_id in enumerate(flat_ids.tolist()):
            shard_index = next(index for index, end in enumerate(shard_ends) if row_id < end)
            grouped[shard_index].append((result_index, row_id - self._shard_offsets[shard_index]))
        for shard_index, pairs in grouped.items():
            key, filename = self._shards[shard_index]
            result_indices, local_rows = zip(*pairs, strict=True)
            with safe_open(str(self.checkpoint_dir / filename), framework="pt", device="cpu") as handle:
                values = handle.get_slice(key)[torch.tensor(local_rows, dtype=torch.long)]
            result[torch.tensor(result_indices, dtype=torch.long)] = values
        return result.reshape(*row_ids.shape, self.head_dim)

    def lookup(self, input_ids: torch.Tensor, history: HostNGramHistory) -> torch.Tensor:
        """Lookup `[batch, seq]` IDs then advance host history after the lookup."""
        ids = self.ngram_ids(input_ids, history)
        embeddings = self._lookup_rows(ids).flatten(-2)
        history.update(input_ids)
        return embeddings
