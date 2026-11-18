# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
from typing import Dict, List, Tuple

import datasets
import torch
import torch.distributed as dist
from transformers.data import DataCollatorForSeq2Seq

from QEfficient.finetune.data.sampler import DistributedLengthBasedBatchSampler
from QEfficient.finetune.dataset.dataset_config import DATALOADER_COLLATE_FUNC, DATASET_PREPROC
from QEfficient.finetune.utils.helper import get_world_size
from QEfficient.finetune.utils.logging_utils import logger


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", context_length: int = None
) -> torch.utils.data.Dataset:
    if dataset_config.dataset not in DATASET_PREPROC:
        logger.raise_error(f"{dataset_config.dataset} is not (yet) implemented", NotImplementedError)

    def get_split():
        return dataset_config.train_split if split == "train" else dataset_config.test_split

    return DATASET_PREPROC[dataset_config.dataset](dataset_config, tokenizer, get_split(), context_length)


def get_custom_data_collator(dataset_processer, dataset_config) -> torch.utils.data.Dataset:
    if dataset_config.dataset not in DATALOADER_COLLATE_FUNC:
        return None

    return DATALOADER_COLLATE_FUNC[dataset_config.dataset](dataset_processer, dataset_config)


def get_dataloader_kwargs(train_config, dataset, dataset_processer, split):
    kwargs = {}
    batch_size = train_config.train_batch_size if split == "train" else train_config.val_batch_size
    if train_config.enable_ddp:
        if train_config.enable_sorting_for_ddp:
            if train_config.context_length:
                logger.raise_error(
                    "Sorting cannot be done with padding, Please disable sorting or pass context_length as None to disable padding",
                    ValueError,
                )
            else:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=False,
                )
        else:
            kwargs["sampler"] = torch.utils.data.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = False
    else:
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
    # todo: -100 should be changed to a variable. or tokenizer.pad_token_id
    kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer, label_pad_token_id=-100)
    return kwargs


def padding_dataset(train_config, dataset, batch_size):
    num_replicas = get_world_size()
    remainder = len(dataset) % (num_replicas * batch_size)
    if remainder == 0:
        return dataset

    if train_config.enable_ddp and train_config.enable_sorting_for_ddp:
        if isinstance(dataset, datasets.Dataset):
            # Hugging Face Dataset transformation
            dataset = dataset.map(lambda x: {"input_length": len(x["input_ids"])})
            dataset = dataset.sort("input_length")
            dataset = dataset.remove_columns("input_length")
        else:
            dataset = sorted(dataset, key=lambda x: len(x["input_ids"]))

    dummy_row = next(iter(dataset))
    dummy_row["labels"] = torch.tensor([-100] * len(dummy_row["labels"]))

    padding_size = (num_replicas * batch_size) - remainder
    dummy_data = [dummy_row.copy() for _ in range(padding_size)]
    dummy_dataset = datasets.Dataset.from_list(dummy_data)
    if isinstance(dataset, datasets.Dataset):
        combined_dataset = datasets.concatenate_datasets([dataset, dummy_dataset])
    else:
        combined_dataset = dataset + list(dummy_dataset)

    logger.log_rank_zero("Padding dataset to make it divisible by batch_size * num_devices.", logging.DEBUG)
    logger.log_rank_zero(f"Length of dataset before padding: {len(dataset)}", logging.DEBUG)
    logger.log_rank_zero(f"Length of dataset after padding: {len(combined_dataset)}", logging.DEBUG)
    return combined_dataset


def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split, context_length=train_config.context_length)

    batch_size = train_config.train_batch_size if split == "train" else train_config.val_batch_size

    dataset = padding_dataset(train_config, dataset, batch_size)

    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)

    # FIXME (Meet): Add custom data collator registration from the outside by the user.
    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)

    if custom_data_collator:
        print("custom_data_collator is used")
        dl_kwargs["collate_fn"] = custom_data_collator

    logger.log_rank_zero(f"Length of {split} dataset is {len(dataset)}")

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    def get_input_lengths(dataset):
        if isinstance(dataset, datasets.Dataset):
            return [len(input_ids) for input_ids in dataset.with_format(None)["input_ids"]]
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            return [length for child_dataset in dataset.datasets for length in get_input_lengths(child_dataset)]
        return [len(sample["input_ids"]) for sample in dataset]

    lengths = get_input_lengths(data)
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix
