# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
from chatterbox.models.s3gen.utils.mask import add_optional_chunk_mask, make_pad_mask
from torch import nn
from torch.nn.utils import parametrize


def fold_weight_norm(module: nn.Module) -> None:
    """Fold weight_norm parametrizations into plain weights, in place.

    Never call it on a deepcopy of a module that is still in use: the copy shares the dynamically created
    ``Parametrized*`` class, so removing the parametrization from the copy also breaks the original.
    """
    for submodule in list(module.modules()):  # collect first: removing a parametrization mutates the tree
        if parametrize.is_parametrized(submodule):
            for name in list(submodule.parametrizations.keys()):
                parametrize.remove_parametrizations(submodule, name, leave_parametrized=True)


class QEffUpsampleConformerEncoder(UpsampleConformerEncoder):
    """S3Gen flow encoder that gives the unpadded result on length-padded (bucketed) inputs.

    Copied from UpsampleConformerEncoder: https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/s3gen/transformer/upsample_encoder.py
    One line added (``xs = xs * masks...``): padded positions are zeroed before the 3-token look-ahead convolution,
    which is what the unpadded upstream run sees past the end of the sequence (its ``F.pad`` zeros).
    """

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        xs = xs * masks.transpose(1, 2).to(xs.dtype)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )
        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        xs, xs_lens = self.up_layer(xs.transpose(1, 2).contiguous(), xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size * self.up_layer.stride,
            num_decoding_left_chunks,
        )
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


class QEffS3GenFlowEncoderWrapper(nn.Module):
    """Speech tokens (prompt + generated, padded to a bucket) -> flow conditioning ``mu`` and the frame mask."""

    def __init__(self, flow: nn.Module):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj

    def forward(self, token: torch.Tensor, token_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_mask = (~make_pad_mask(token_len, token.shape[1])).unsqueeze(-1).to(self.encoder_proj.weight.dtype)
        hidden, frame_mask = self.encoder(self.input_embedding(token) * token_mask, token_len)
        return self.encoder_proj(hidden).transpose(1, 2), frame_mask.to(hidden.dtype)


class QEffConvRNNF0Predictor(ConvRNNF0Predictor):
    """F0 predictor with weight_norm folded into plain weights for export."""

    def __qeff_init__(self):
        fold_weight_norm(self)


class QEffConvTranspose1d(nn.ConvTranspose1d):
    """1D transposed convolution computed as a height-1 2D one: qaic-compile has no 1D ConvTranspose."""

    def forward(self, input: torch.Tensor, output_size: Optional[list] = None) -> torch.Tensor:
        return F.conv_transpose2d(
            input[:, :, None, :],
            self.weight[:, :, None, :],
            self.bias,
            stride=(1, self.stride[0]),
            padding=(0, self.padding[0]),
            output_padding=(0, self.output_padding[0]),
            groups=self.groups,
            dilation=(1, self.dilation[0]),
        )[:, :, 0, :]


class QEffHiFTGenerator(HiFTGenerator):
    """HiFT vocoder whose STFT / inverse STFT are fixed-kernel convolutions.

    ``torch.stft`` (complex output) and ``torch.istft`` do not export to ONNX. With n_fft=16 and hop=4 both are
    small convolutions with a windowed DFT basis; the inverse divides by the window overlap-add envelope.
    """

    def __qeff_init__(self):
        fold_weight_norm(self)
        n_fft = self.istft_params["n_fft"]
        window = self.stft_window.double()
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        angle = 2 * math.pi * k * n / n_fft
        forward_basis = torch.cat([window * torch.cos(angle), -window * torch.sin(angle)])
        onesided_scale = torch.full((n_fft // 2 + 1, 1), 2.0, dtype=torch.float64)
        onesided_scale[0] = onesided_scale[-1] = 1.0
        inverse_basis = torch.cat([onesided_scale * torch.cos(angle), -onesided_scale * torch.sin(angle)]) / n_fft
        dtype = self.conv_post.weight.dtype
        self.register_buffer("stft_kernel", forward_basis[:, None, :].to(dtype), persistent=False)
        self.register_buffer("istft_kernel", (inverse_basis * window)[:, None, None, :].to(dtype), persistent=False)
        self.register_buffer("istft_window_sq", (window * window)[None, None, None, :].to(dtype), persistent=False)

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        x = F.pad(x[:, None, :], (n_fft // 2, n_fft // 2), mode="reflect")  # torch.stft center=True
        spec = F.conv1d(x, self.stft_kernel, stride=hop)
        return spec[:, : n_fft // 2 + 1], spec[:, n_fft // 2 + 1 :]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        magnitude = torch.clip(magnitude, max=1e2)
        spec = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)[:, :, None, :]
        frames = F.conv_transpose2d(spec, self.istft_kernel, stride=(1, hop))
        envelope = F.conv_transpose2d(torch.ones_like(spec[:, :1]), self.istft_window_sq, stride=(1, hop))
        return (frames / envelope)[:, 0, 0, n_fft // 2 : -(n_fft // 2)]


class QEffHiFTDecodeWrapper(nn.Module):
    """HiFT decode for export: mel + excitation source (computed on the host in fp32) -> waveform."""

    def __init__(self, hift: nn.Module):
        super().__init__()
        self.hift = hift

    def forward(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.hift.decode(x=mel, s=source)
