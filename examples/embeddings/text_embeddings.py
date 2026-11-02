# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


def main():
    parser = argparse.ArgumentParser(description="Text embeddings inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model ID",
    )
    parser.add_argument(
        "--sentences",
        type=str,
        default="This is an example sentence",
        help="Input sentence(s) to generate embeddings for",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean", "none"],
        help="Pooling strategy: 'max' for max pooling, 'mean' for mean pooling, 'none' for no pooling",
    )
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--seq-len",
        type=str,
        default="32,64",
        help="Sequence length(s) - single int (e.g., '32') or comma-separated list (e.g., '32,64')",
    )
    args = parser.parse_args()

    # Parse seq_len argument
    if "," in args.seq_len:
        seq_len = [int(x.strip()) for x in args.seq_len.split(",")]
    else:
        seq_len = int(args.seq_len)

    print(f"Loading embedding model: {args.model_name}")
    print(f"Pooling strategy: {args.pooling}")
    print(f"Sequence length(s): {seq_len}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model with pooling strategy
    # You can specify the pooling strategy either as a string (e.g., "max") or by passing a custom pooling function.
    # If no pooling is specified, the model will return its default output (typically token embeddings).
    if args.pooling == "max":
        qeff_model = AutoModel.from_pretrained(args.model_name, pooling=max_pooling)
    elif args.pooling == "mean":
        qeff_model = AutoModel.from_pretrained(args.model_name, pooling="mean")
    else:
        qeff_model = AutoModel.from_pretrained(args.model_name)

    # Compile the model
    # seq_len can be a list of seq_len or single int
    qeff_model.compile(num_cores=args.num_cores, seq_len=seq_len)

    # Tokenize sentences
    encoded_input = tokenizer(args.sentences, return_tensors="pt")

    # Run the generation
    sentence_embeddings = qeff_model.generate(encoded_input)

    print(f"\nInput: {args.sentences}")
    print(f"Sentence embeddings shape: {sentence_embeddings['output'].shape}")
    print(f"Sentence embeddings: {sentence_embeddings}")


if __name__ == "__main__":
    main()
