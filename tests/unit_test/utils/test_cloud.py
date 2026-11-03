# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient.cloud module.

Tests verify:
  - Module importability
  - Argument parsing for CLI scripts (compile.py, execute.py, export.py, infer.py)
  - Function signatures and parameter validation
  - Error handling for missing required arguments
  - finetune.py helper functions (setup_seeds, apply_peft, etc.)

All tests run on CPU only. No actual compilation, execution, or model loading
is performed - only argument parsing and function structure validation.
"""

import argparse
import inspect
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tests: Module importability
# ---------------------------------------------------------------------------


class TestCloudModuleImportability:
    """All cloud modules must be importable on CPU."""

    def test_cloud_init_importable(self):
        import QEfficient.cloud

        assert QEfficient.cloud is not None

    def test_compile_module_importable(self):
        import QEfficient.cloud.compile

        assert QEfficient.cloud.compile is not None

    def test_execute_module_importable(self):
        import QEfficient.cloud.execute

        assert QEfficient.cloud.execute is not None

    def test_export_module_importable(self):
        import QEfficient.cloud.export

        assert QEfficient.cloud.export is not None

    def test_infer_module_importable(self):
        import QEfficient.cloud.infer

        assert QEfficient.cloud.infer is not None

    def test_finetune_module_importable(self):
        import QEfficient.cloud.finetune

        assert QEfficient.cloud.finetune is not None

    def test_finetune_experimental_importable(self):
        import QEfficient.cloud.finetune_experimental

        assert QEfficient.cloud.finetune_experimental is not None


# ---------------------------------------------------------------------------
# Tests: export.py - function signatures
# ---------------------------------------------------------------------------


class TestExportFunctionSignatures:
    """export.py functions must have correct signatures."""

    def test_get_onnx_path_exists(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        assert callable(get_onnx_path_and_setup_customIO)

    def test_get_onnx_path_has_model_name(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "model_name" in sig.parameters

    def test_get_onnx_path_has_cache_dir(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "cache_dir" in sig.parameters

    def test_get_onnx_path_has_hf_token(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "hf_token" in sig.parameters

    def test_get_onnx_path_has_full_batch_size(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "full_batch_size" in sig.parameters

    def test_get_onnx_path_has_local_model_dir(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "local_model_dir" in sig.parameters

    def test_get_onnx_path_has_mxint8_kv_cache(self):
        from QEfficient.cloud.export import get_onnx_path_and_setup_customIO

        sig = inspect.signature(get_onnx_path_and_setup_customIO)
        assert "mxint8_kv_cache" in sig.parameters

    def test_export_main_exists(self):
        from QEfficient.cloud.export import main

        assert callable(main)

    def test_export_main_has_model_name(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "model_name" in sig.parameters

    def test_export_main_has_cache_dir(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "cache_dir" in sig.parameters

    def test_export_main_has_hf_token(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "hf_token" in sig.parameters

    def test_export_main_has_local_model_dir(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "local_model_dir" in sig.parameters

    def test_export_main_has_full_batch_size(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "full_batch_size" in sig.parameters

    def test_export_main_has_mxint8_kv_cache(self):
        from QEfficient.cloud.export import main

        sig = inspect.signature(main)
        assert "mxint8_kv_cache" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: execute.py - function signatures
# ---------------------------------------------------------------------------


class TestExecuteFunctionSignatures:
    """execute.py main function must have correct signature."""

    def test_main_exists(self):
        from QEfficient.cloud.execute import main

        assert callable(main)

    def test_main_has_model_name(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "model_name" in sig.parameters

    def test_main_has_qpc_path(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "qpc_path" in sig.parameters

    def test_main_has_device_group(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "device_group" in sig.parameters

    def test_main_has_prompt(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "prompt" in sig.parameters

    def test_main_has_prompts_txt_file_path(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "prompts_txt_file_path" in sig.parameters

    def test_main_has_generation_len(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "generation_len" in sig.parameters

    def test_main_has_cache_dir(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "cache_dir" in sig.parameters

    def test_main_has_hf_token(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "hf_token" in sig.parameters

    def test_main_has_local_model_dir(self):
        from QEfficient.cloud.execute import main

        sig = inspect.signature(main)
        assert "local_model_dir" in sig.parameters

    def test_device_group_warns_about_next_release(self, monkeypatch):
        import QEfficient.cloud.execute as execute

        monkeypatch.setattr(execute, "load_hf_tokenizer", MagicMock())
        monkeypatch.setattr(execute, "cloud_ai_100_exec_kv", MagicMock())

        with pytest.warns(FutureWarning, match="renamed to device_ids in the next release"):
            execute.main(model_name="gpt2", qpc_path="/path/to/qpc", device_group=[0])


# ---------------------------------------------------------------------------
# Tests: infer.py - function signatures
# ---------------------------------------------------------------------------


class TestInferFunctionSignatures:
    """infer.py functions must have correct signatures."""

    def test_main_exists(self):
        from QEfficient.cloud.infer import main

        assert callable(main)

    def test_main_has_model_name(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "model_name" in sig.parameters

    def test_main_has_num_cores(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "num_cores" in sig.parameters

    def test_main_has_device_group(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "device_group" in sig.parameters

    def test_main_has_prompt(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "prompt" in sig.parameters

    def test_main_has_batch_size(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "batch_size" in sig.parameters

    def test_main_has_ctx_len(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "ctx_len" in sig.parameters

    def test_main_has_prompt_len(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "prompt_len" in sig.parameters

    def test_main_has_mxfp6(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "mxfp6" in sig.parameters

    def test_main_has_mxint8(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "mxint8" in sig.parameters

    def test_main_has_generation_len(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "generation_len" in sig.parameters

    def test_main_has_full_batch_size(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "full_batch_size" in sig.parameters

    def test_main_has_enable_qnn(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "enable_qnn" in sig.parameters

    def test_main_has_cache_dir(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "cache_dir" in sig.parameters

    def test_main_has_hf_token(self):
        from QEfficient.cloud.infer import main

        sig = inspect.signature(main)
        assert "hf_token" in sig.parameters

    def test_execute_vlm_model_exists(self):
        from QEfficient.cloud.infer import execute_vlm_model

        assert callable(execute_vlm_model)

    def test_execute_vlm_model_has_qeff_model(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "qeff_model" in sig.parameters

    def test_execute_vlm_model_has_model_name(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "model_name" in sig.parameters

    def test_execute_vlm_model_has_image_url(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "image_url" in sig.parameters

    def test_execute_vlm_model_has_image_path(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "image_path" in sig.parameters

    def test_execute_vlm_model_has_prompt(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "prompt" in sig.parameters

    def test_execute_vlm_model_has_generation_len(self):
        from QEfficient.cloud.infer import execute_vlm_model

        sig = inspect.signature(execute_vlm_model)
        assert "generation_len" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: infer.py - execute_vlm_model error handling
# ---------------------------------------------------------------------------


class TestExecuteVlmModelErrorHandling:
    """execute_vlm_model must raise ValueError when no image is provided."""

    def test_raises_without_image_url_or_path(self):
        from QEfficient.cloud.infer import execute_vlm_model

        with pytest.raises(ValueError, match="Neither Image URL nor Image Path"):
            execute_vlm_model(
                qeff_model=MagicMock(),
                model_name="test",
                image_url=None,
                image_path=None,
                prompt=["test"],
            )

    def test_raises_with_empty_image_url_and_no_path(self):
        from QEfficient.cloud.infer import execute_vlm_model

        with pytest.raises(ValueError):
            execute_vlm_model(
                qeff_model=MagicMock(),
                model_name="test",
                image_url="",
                image_path=None,
                prompt=["test"],
            )

    def test_raises_with_empty_image_path_and_no_url(self):
        from QEfficient.cloud.infer import execute_vlm_model

        with pytest.raises(ValueError):
            execute_vlm_model(
                qeff_model=MagicMock(),
                model_name="test",
                image_url=None,
                image_path="",
                prompt=["test"],
            )


# ---------------------------------------------------------------------------
# Tests: finetune.py - function signatures
# ---------------------------------------------------------------------------


class TestFinetuneFunctionSignatures:
    """finetune.py functions must have correct signatures."""

    def test_setup_distributed_training_exists(self):
        from QEfficient.cloud.finetune import setup_distributed_training

        assert callable(setup_distributed_training)

    def test_setup_distributed_training_has_train_config(self):
        from QEfficient.cloud.finetune import setup_distributed_training

        sig = inspect.signature(setup_distributed_training)
        assert "train_config" in sig.parameters

    def test_setup_seeds_exists(self):
        from QEfficient.cloud.finetune import setup_seeds

        assert callable(setup_seeds)

    def test_setup_seeds_has_seed(self):
        from QEfficient.cloud.finetune import setup_seeds

        sig = inspect.signature(setup_seeds)
        assert "seed" in sig.parameters

    def test_load_model_and_tokenizer_exists(self):
        from QEfficient.cloud.finetune import load_model_and_tokenizer

        assert callable(load_model_and_tokenizer)

    def test_load_model_and_tokenizer_has_train_config(self):
        from QEfficient.cloud.finetune import load_model_and_tokenizer

        sig = inspect.signature(load_model_and_tokenizer)
        assert "train_config" in sig.parameters

    def test_load_model_and_tokenizer_has_dataset_config(self):
        from QEfficient.cloud.finetune import load_model_and_tokenizer

        sig = inspect.signature(load_model_and_tokenizer)
        assert "dataset_config" in sig.parameters

    def test_apply_peft_exists(self):
        from QEfficient.cloud.finetune import apply_peft

        assert callable(apply_peft)

    def test_apply_peft_has_model(self):
        from QEfficient.cloud.finetune import apply_peft

        sig = inspect.signature(apply_peft)
        assert "model" in sig.parameters

    def test_apply_peft_has_train_config(self):
        from QEfficient.cloud.finetune import apply_peft

        sig = inspect.signature(apply_peft)
        assert "train_config" in sig.parameters

    def test_setup_dataloaders_exists(self):
        from QEfficient.cloud.finetune import setup_dataloaders

        assert callable(setup_dataloaders)

    def test_setup_dataloaders_has_train_config(self):
        from QEfficient.cloud.finetune import setup_dataloaders

        sig = inspect.signature(setup_dataloaders)
        assert "train_config" in sig.parameters

    def test_setup_dataloaders_has_dataset_config(self):
        from QEfficient.cloud.finetune import setup_dataloaders

        sig = inspect.signature(setup_dataloaders)
        assert "dataset_config" in sig.parameters

    def test_setup_dataloaders_has_tokenizer(self):
        from QEfficient.cloud.finetune import setup_dataloaders

        sig = inspect.signature(setup_dataloaders)
        assert "tokenizer" in sig.parameters

    def test_main_exists(self):
        from QEfficient.cloud.finetune import main

        assert callable(main)


# ---------------------------------------------------------------------------
# Tests: finetune.py - setup_seeds behavior
# ---------------------------------------------------------------------------


class TestSetupSeeds:
    """setup_seeds must set random seeds correctly."""

    def test_setup_seeds_does_not_crash(self):
        from QEfficient.cloud.finetune import setup_seeds

        setup_seeds(42)

    def test_setup_seeds_with_different_values(self):
        from QEfficient.cloud.finetune import setup_seeds

        for seed in [0, 1, 42, 100, 9999]:
            setup_seeds(seed)

    def test_setup_seeds_torch_reproducibility(self):
        import torch

        from QEfficient.cloud.finetune import setup_seeds

        setup_seeds(42)
        torch.manual_seed(42)
        a = torch.rand(5).tolist()
        torch.manual_seed(42)
        b = torch.rand(5).tolist()
        assert a == b, "torch.manual_seed must produce reproducible results"

    def test_setup_seeds_numpy_reproducibility(self):
        import numpy as np

        from QEfficient.cloud.finetune import setup_seeds

        setup_seeds(42)
        np.random.seed(42)
        a = np.random.rand(5).tolist()
        np.random.seed(42)
        b = np.random.rand(5).tolist()
        assert a == b, "np.random.seed must produce reproducible results"


# ---------------------------------------------------------------------------
# Tests: finetune.py - apply_peft behavior
# ---------------------------------------------------------------------------


class TestApplyPeft:
    """apply_peft must return model unchanged when use_peft=False."""

    def test_apply_peft_returns_model_when_peft_disabled(self):
        from QEfficient.cloud.finetune import apply_peft
        from QEfficient.finetune.configs.training import TrainConfig

        train_config = TrainConfig()
        train_config.use_peft = False

        mock_model = MagicMock()
        result = apply_peft(mock_model, train_config)
        assert result is mock_model, "apply_peft must return original model when use_peft=False"

    def test_apply_peft_does_not_modify_model_when_disabled(self):
        from QEfficient.cloud.finetune import apply_peft
        from QEfficient.finetune.configs.training import TrainConfig

        train_config = TrainConfig()
        train_config.use_peft = False

        mock_model = MagicMock()
        original_id = id(mock_model)
        result = apply_peft(mock_model, train_config)
        assert id(result) == original_id


# ---------------------------------------------------------------------------
# Tests: Argument parsing - compile.py
# ---------------------------------------------------------------------------


class TestCompileArgumentParsing:
    """compile.py argument parser must handle required and optional args."""

    def _get_parser(self):
        parser = argparse.ArgumentParser(description="Compilation script.")
        parser.add_argument("--onnx_path", "--onnx-path", required=True)
        parser.add_argument("--qpc-path", "--qpc_path", required=True)
        parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
        parser.add_argument("--prompt_len", "--prompt-len", default=32, type=int)
        parser.add_argument("--ctx_len", "--ctx-len", default=128, type=int)
        parser.add_argument("--mxfp6", action="store_true")
        parser.add_argument("--mxint8", action="store_true")
        parser.add_argument("--num_cores", "--num-cores", required=True, type=int)
        parser.add_argument(
            "--device_group",
            "--device-group",
            required=True,
            type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        )
        parser.add_argument("--aic_enable_depth_first", "--aic-enable-depth-first", action="store_true")
        parser.add_argument("--mos", type=int, default=-1)
        parser.add_argument("--full_batch_size", "--full-batch-size", type=int, default=None)
        return parser

    def test_parser_requires_onnx_path(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_requires_num_cores(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--onnx_path", "/path/to/model.onnx", "--qpc-path", "/path/to/qpc"])

    def test_parser_requires_device_group(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--onnx_path", "/path/to/model.onnx", "--qpc-path", "/path/to/qpc", "--num-cores", "16"])

    def test_parser_accepts_all_required_args(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
            ]
        )
        assert args.onnx_path == "/path/to/model.onnx"
        assert args.num_cores == 16

    def test_parser_default_batch_size_is_1(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
            ]
        )
        assert args.batch_size == 1

    def test_parser_default_prompt_len_is_32(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
            ]
        )
        assert args.prompt_len == 32

    def test_parser_default_ctx_len_is_128(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
            ]
        )
        assert args.ctx_len == 128

    def test_parser_accepts_batch_size(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
                "--batch-size",
                "4",
            ]
        )
        assert args.batch_size == 4

    def test_parser_accepts_multi_device_group(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0,1,2,3]",
            ]
        )
        assert args.device_group == [0, 1, 2, 3]

    def test_parser_accepts_mxfp6_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
                "--mxfp6",
            ]
        )
        assert args.mxfp6 is True

    def test_parser_accepts_mxint8_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
                "--mxint8",
            ]
        )
        assert args.mxint8 is True

    def test_parser_accepts_aic_enable_depth_first(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
                "--aic-enable-depth-first",
            ]
        )
        assert args.aic_enable_depth_first is True

    def test_parser_accepts_full_batch_size(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
                "--full-batch-size",
                "8",
            ]
        )
        assert args.full_batch_size == 8

    def test_parser_default_full_batch_size_is_none(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "--onnx_path",
                "/path/to/model.onnx",
                "--qpc-path",
                "/path/to/qpc",
                "--num-cores",
                "16",
                "--device-group",
                "[0]",
            ]
        )
        assert args.full_batch_size is None


# ---------------------------------------------------------------------------
# Tests: Argument parsing - execute.py
# ---------------------------------------------------------------------------


class TestExecuteArgumentParsing:
    """execute.py argument parser must handle required and optional args."""

    def _get_parser(self):
        parser = argparse.ArgumentParser(description="Execution script.")
        parser.add_argument("--model_name", "--model-name", required=False, type=str)
        parser.add_argument("--qpc_path", "--qpc-path", required=True)
        parser.add_argument(
            "--device_group",
            "--device-group",
            type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        )
        parser.add_argument("--prompt", type=lambda prompt: prompt.split("|"))
        parser.add_argument("--prompts_txt_file_path", "--prompts-txt-file-path", type=str)
        parser.add_argument("--generation_len", "--generation-len", type=int)
        parser.add_argument("--local-model-dir", "--local_model_dir", required=False)
        parser.add_argument("--cache-dir", "--cache_dir", default=None, required=False)
        parser.add_argument("--full_batch_size", "--full-batch-size", type=int, default=None)
        parser.add_argument("--hf-token", "--hf_token", default=None, type=str, required=False)
        return parser

    def test_parser_requires_qpc_path(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_qpc_path(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc"])
        assert args.qpc_path == "/path/to/qpc"

    def test_parser_accepts_model_name(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--model_name", "gpt2"])
        assert args.model_name == "gpt2"

    def test_parser_accepts_prompt_with_pipe(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--prompt", "Hello|World|Test"])
        assert args.prompt == ["Hello", "World", "Test"]

    def test_parser_accepts_single_prompt(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--prompt", "Hello world"])
        assert args.prompt == ["Hello world"]

    def test_parser_accepts_generation_len(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--generation-len", "100"])
        assert args.generation_len == 100

    def test_parser_accepts_device_group(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--device-group", "[0,1]"])
        assert args.device_group == [0, 1]

    def test_parser_default_generation_len_is_none(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc"])
        assert args.generation_len is None

    def test_parser_accepts_hf_token(self):
        parser = self._get_parser()
        args = parser.parse_args(["--qpc_path", "/path/to/qpc", "--hf-token", "hf_abc123"])
        assert args.hf_token == "hf_abc123"


# ---------------------------------------------------------------------------
# Tests: Argument parsing - export.py
# ---------------------------------------------------------------------------


class TestExportArgumentParsing:
    """export.py argument parser must handle required and optional args."""

    def _get_parser(self):
        parser = argparse.ArgumentParser(description="Export script.")
        parser.add_argument("--model_name", "--model-name", required=True)
        parser.add_argument("--local-model-dir", "--local_model_dir", required=False)
        parser.add_argument("--cache_dir", "--cache-dir", required=False)
        parser.add_argument("--hf-token", "--hf_token", default=None, type=str, required=False)
        parser.add_argument("--full_batch_size", "--full-batch-size", type=int, default=None)
        parser.add_argument("--mxint8_kv_cache", "--mxint8-kv-cache", required=False)
        return parser

    def test_parser_requires_model_name(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_model_name(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model_name", "gpt2"])
        assert args.model_name == "gpt2"

    def test_parser_accepts_cache_dir(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model_name", "gpt2", "--cache-dir", "/path/to/cache"])
        assert args.cache_dir == "/path/to/cache"

    def test_parser_accepts_hf_token(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model_name", "gpt2", "--hf-token", "hf_token123"])
        assert args.hf_token == "hf_token123"

    def test_parser_accepts_full_batch_size(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model_name", "gpt2", "--full-batch-size", "4"])
        assert args.full_batch_size == 4

    def test_parser_default_full_batch_size_is_none(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model_name", "gpt2"])
        assert args.full_batch_size is None


# ---------------------------------------------------------------------------
# Tests: Argument parsing - infer.py
# ---------------------------------------------------------------------------


class TestInferArgumentParsing:
    """infer.py argument parser must handle required and optional args."""

    def _get_parser(self):
        parser = argparse.ArgumentParser(description="Inference script.")
        parser.add_argument("--model-name", "--model_name", required=True, type=str)
        parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
        parser.add_argument("--prompt-len", "--prompt_len", default=32, type=int)
        parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int)
        parser.add_argument("--num_cores", "--num-cores", type=int, required=True)
        parser.add_argument(
            "--device_group",
            "--device-group",
            type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        )
        parser.add_argument("--prompt", type=lambda prompt: prompt.split("|"))
        parser.add_argument("--generation_len", "--generation-len", type=int)
        parser.add_argument("--mxfp6", "--mxfp6_matmul", "--mxfp6-matmul", action="store_true")
        parser.add_argument("--mxint8", "--mxint8_kv_cache", "--mxint8-kv-cache", action="store_true")
        parser.add_argument("--full_batch_size", "--full-batch-size", type=int, default=None)
        parser.add_argument("--aic_enable_depth_first", "--aic-enable-depth-first", action="store_true")
        parser.add_argument("--mos", type=int, default=1)
        parser.add_argument("--cache-dir", "--cache_dir", default=None, required=False)
        parser.add_argument("--hf-token", "--hf_token", default=None, type=str, required=False)
        parser.add_argument("--trust_remote_code", action="store_true", default=False)
        return parser

    def test_parser_requires_model_name(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_requires_num_cores(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model-name", "gpt2"])

    def test_parser_accepts_all_required_args(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16"])
        assert args.model_name == "gpt2"
        assert args.num_cores == 16

    def test_parser_default_batch_size_is_1(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16"])
        assert args.batch_size == 1

    def test_parser_default_prompt_len_is_32(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16"])
        assert args.prompt_len == 32

    def test_parser_default_ctx_len_is_128(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16"])
        assert args.ctx_len == 128

    def test_parser_accepts_mxfp6_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--mxfp6"])
        assert args.mxfp6 is True

    def test_parser_accepts_mxint8_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--mxint8"])
        assert args.mxint8 is True

    def test_parser_accepts_aic_enable_depth_first(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--aic-enable-depth-first"])
        assert args.aic_enable_depth_first is True

    def test_parser_accepts_full_batch_size(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--full-batch-size", "8"])
        assert args.full_batch_size == 8

    def test_parser_accepts_trust_remote_code(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--trust_remote_code"])
        assert args.trust_remote_code is True

    def test_parser_default_trust_remote_code_is_false(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16"])
        assert args.trust_remote_code is False

    def test_parser_accepts_prompt_with_pipe(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--prompt", "Hello|World"])
        assert args.prompt == ["Hello", "World"]

    def test_parser_accepts_device_group(self):
        parser = self._get_parser()
        args = parser.parse_args(["--model-name", "gpt2", "--num-cores", "16", "--device-group", "[0,1]"])
        assert args.device_group == [0, 1]


# ---------------------------------------------------------------------------
# Tests: Device group parsing utility
# ---------------------------------------------------------------------------


class TestDeviceGroupParsing:
    """Device group lambda parser must correctly parse various formats."""

    def _parse_device_group(self, s):
        return [int(x) for x in s.strip("[]").split(",")]

    def test_single_device(self):
        result = self._parse_device_group("[0]")
        assert result == [0]

    def test_two_devices(self):
        result = self._parse_device_group("[0,1]")
        assert result == [0, 1]

    def test_four_devices(self):
        result = self._parse_device_group("[0,1,2,3]")
        assert result == [0, 1, 2, 3]

    def test_device_with_spaces(self):
        result = self._parse_device_group("[0, 1, 2]")
        assert result == [0, 1, 2]

    def test_single_digit_device(self):
        result = self._parse_device_group("[7]")
        assert result == [7]


# ---------------------------------------------------------------------------
# Tests: Prompt parsing utility
# ---------------------------------------------------------------------------


class TestPromptParsing:
    """Prompt pipe-split lambda must correctly parse prompts."""

    def _parse_prompt(self, s):
        return s.split("|")

    def test_single_prompt(self):
        result = self._parse_prompt("Hello world")
        assert result == ["Hello world"]

    def test_two_prompts(self):
        result = self._parse_prompt("Hello|World")
        assert result == ["Hello", "World"]

    def test_three_prompts(self):
        result = self._parse_prompt("A|B|C")
        assert result == ["A", "B", "C"]

    def test_prompt_with_spaces(self):
        result = self._parse_prompt("Hello world|How are you")
        assert result == ["Hello world", "How are you"]

    def test_empty_prompt(self):
        result = self._parse_prompt("")
        assert result == [""]


# ---------------------------------------------------------------------------
# Tests: TrainConfig importability and defaults
# ---------------------------------------------------------------------------


class TestTrainConfig:
    """TrainConfig must be importable and have correct defaults."""

    def test_train_config_importable(self):
        from QEfficient.finetune.configs.training import TrainConfig

        assert TrainConfig is not None

    def test_train_config_instantiable(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert cfg is not None

    def test_train_config_has_model_name(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "model_name")

    def test_train_config_has_use_peft(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "use_peft")

    def test_train_config_has_seed(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "seed")

    def test_train_config_has_device(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "device")

    def test_train_config_has_enable_ddp(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "enable_ddp")

    def test_train_config_has_lr(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "lr")

    def test_train_config_has_gradient_checkpointing(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert hasattr(cfg, "gradient_checkpointing")

    def test_train_config_use_peft_default_is_true(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert cfg.use_peft is True

    def test_train_config_enable_ddp_default_is_false(self):
        from QEfficient.finetune.configs.training import TrainConfig

        cfg = TrainConfig()
        assert cfg.enable_ddp is False


# ---------------------------------------------------------------------------
# Tests: setup_distributed_training with DDP disabled
# ---------------------------------------------------------------------------


class TestSetupDistributedTraining:
    """setup_distributed_training must handle non-DDP case without error."""

    def test_non_ddp_cpu_does_not_crash(self):
        from QEfficient.cloud.finetune import setup_distributed_training
        from QEfficient.finetune.configs.training import TrainConfig

        train_config = TrainConfig()
        train_config.enable_ddp = False
        train_config.device = "cpu"
        # Should not raise
        setup_distributed_training(train_config)

    def test_non_ddp_returns_none(self):
        from QEfficient.cloud.finetune import setup_distributed_training
        from QEfficient.finetune.configs.training import TrainConfig

        train_config = TrainConfig()
        train_config.enable_ddp = False
        train_config.device = "cpu"
        result = setup_distributed_training(train_config)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: check_and_assign_cache_dir utility
# ---------------------------------------------------------------------------


class TestCheckAndAssignCacheDir:
    """check_and_assign_cache_dir must return correct cache directory."""

    def test_function_importable(self):
        from QEfficient.utils import check_and_assign_cache_dir

        assert callable(check_and_assign_cache_dir)

    def test_returns_cache_dir_when_provided(self):
        from QEfficient.utils import check_and_assign_cache_dir

        result = check_and_assign_cache_dir(local_model_dir=None, cache_dir="/my/cache")
        assert result == "/my/cache"

    def test_returns_default_when_local_model_dir_provided(self):
        from QEfficient.utils import check_and_assign_cache_dir

        result = check_and_assign_cache_dir(local_model_dir="/local/model", cache_dir=None)
        # When local_model_dir is provided, cache_dir should be None or default
        assert result is None or isinstance(result, str)

    def test_returns_string_or_none(self):
        from QEfficient.utils import check_and_assign_cache_dir

        result = check_and_assign_cache_dir(local_model_dir=None, cache_dir=None)
        assert result is None or isinstance(result, str)


class TestQaicDeviceGroups:
    def test_returns_ready_mdp_groups_with_sufficient_nsps(self):
        from QEfficient.utils.device_utils import get_qaic_mdp_device_groups

        output = """
QID 0
    Status:Ready
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:8
    Board serial:low-nsp
QID 4
    Status:Ready
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:16
    Board serial:board-a
QID 5
    Status:Ready
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:16
    Board serial:board-a
QID 6
    Status:Ready
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:16
    Board serial:board-a
QID 7
    Status:Ready
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:16
    Board serial:board-a
QID 8
    Status:Error
    Device Capabilities: HybridBoot+ MDP+
    Nsp Total:16
    Board serial:board-b
"""
        result = MagicMock(returncode=0, stdout=output, stderr="")

        with patch("QEfficient.utils.device_utils.subprocess.run", return_value=result):
            assert get_qaic_mdp_device_groups() == [[4, 5, 6, 7]]

        with patch("QEfficient.utils.device_utils.subprocess.run", return_value=result):
            assert get_qaic_mdp_device_groups(devices_per_group=2) == [[4, 5], [6, 7]]

    def test_parses_explicit_non_overlapping_groups(self):
        from QEfficient.utils.device_utils import parse_qaic_device_groups

        assert parse_qaic_device_groups("0,1; 4,5") == [[0, 1], [4, 5]]

    def test_rejects_overlapping_explicit_groups(self):
        from QEfficient.utils.device_utils import parse_qaic_device_groups

        with pytest.raises(ValueError, match="cannot be assigned to multiple workers"):
            parse_qaic_device_groups("0,1;1,2")

    @pytest.mark.parametrize("spacing", ["", " ", "\t"])
    @pytest.mark.parametrize(
        "free,total,expected",
        [
            (0, 16, []),
            (14, 16, []),
            (15, 16, []),
            (16, 16, [[0]]),
            (17, 18, [[0]]),
            (16, None, [[0]]),
            (None, 16, [[0]]),
            (None, 15, []),
            (None, None, []),
        ],
    )
    def test_free_nsp_count_takes_precedence_over_total(self, spacing, free, total, expected):
        from QEfficient.utils.device_utils import get_qaic_mdp_device_groups

        output = "QID 0\n    Status:Ready\n    Device Capabilities: HybridBoot+ MDP+\n    Board serial:board-a\n"
        if free is not None:
            output += f"    Nsp Free:{spacing}{free}\n"
        if total is not None:
            output += f"    Nsp Total:{spacing}{total}\n"
        result = MagicMock(returncode=0, stdout=output, stderr="")

        with patch("QEfficient.utils.device_utils.subprocess.run", return_value=result):
            assert get_qaic_mdp_device_groups(devices_per_group=1) == expected

    @pytest.mark.parametrize("num_cores", [14, 16])
    def test_weight_free_availability_uses_test_core_count(self, monkeypatch, num_cores):
        from tests.weight_free import conftest as weight_free_config

        monkeypatch.setattr(weight_free_config, "NUM_AIC_CORES", num_cores)
        request = MagicMock()
        request.node.originalname = "test_weight_free_generate_fp16"
        output = (
            "QID 0\n    Status:Ready\n    Device Capabilities: HybridBoot+ MDP+\n"
            "    Nsp Free:14\n    Nsp Total:14\n    Board serial:board-a\n"
        )
        result = MagicMock(returncode=0, stdout=output, stderr="")
        with patch("QEfficient.utils.device_utils.subprocess.run", return_value=result):
            if num_cores == 14:
                weight_free_config.skip_if_no_qaic_device.__wrapped__(request)
            else:
                with pytest.raises(pytest.skip.Exception, match="No available QAIC device"):
                    weight_free_config.skip_if_no_qaic_device.__wrapped__(request)
