# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import QEfficient.utils.logging_utils as logging_utils
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.utils.logging_utils import (
    LoggerConfig,
    QEFFLogger,
    log_api_arguments,
    log_generate_call,
    log_pipeline_api,
)


@pytest.fixture(autouse=True)
def reset_logger_state():
    QEFFLogger.close_logger()
    yield
    QEFFLogger.close_logger()
    # Keep process-global logger initialized for tests importing module-level adapters.
    QEFFLogger.get_logger("INFRA")


def test_structured_api_and_generation_logging(tmp_path):
    class TestModel:
        @log_generate_call
        def generate(self, generation_len=4):
            return generation_len * 2

    QEFFLogger.get_logger("test", "INFO", str(tmp_path))
    log_api_arguments("compile", "QEFFTestModel", {"token": "secret", "path": Path("/tmp/model")})
    assert TestModel().generate() == 8
    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    QEFFLogger.close_logger()

    with open(log_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    compile_record = next(record for record in records if record.get("api") == "compile")
    generate_record = next(record for record in records if record.get("api") == "generate")
    assert compile_record["arguments"] == {"token": "<redacted>", "path": "/tmp/model"}
    assert generate_record["arguments"] == {"generation_len": 4}
    assert any(record["message"] == "Generation completed." for record in records)


def test_warning_once_logs_a_message_once(tmp_path):
    logger = QEFFLogger.get_logger("MODEL", "INFO", str(tmp_path))
    logger.warning_once("cache is disabled for %s", "training")
    logger.warning_once("cache is disabled for %s", "training")
    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    QEFFLogger.close_logger()

    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.startswith("{")]
    warnings = [record for record in records if record["message"] == "cache is disabled for training"]
    assert len(warnings) == 1


def test_complete_timing_table(tmp_path, capsys):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    QEFFLogger.start_run("test-model")
    QEFFLogger.log_event("milestone", "MODEL", "load", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "export", milestone=QEFFLogger.MILESTONE_EXPORT_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "compile", milestone=QEFFLogger.MILESTONE_COMPILE_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "generate", milestone=QEFFLogger.MILESTONE_GENERATION_COMPLETE)

    assert QEFFLogger.print_table() is True
    output = capsys.readouterr().out
    assert "Model Loading" in output
    assert "Model Exporting" in output
    assert "Model Compilation" in output
    assert "Text Generation" in output
    assert "Total Time" in output


def test_cached_stages_are_zero_in_timing_table(tmp_path):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    QEFFLogger.start_run("cached-model")
    QEFFLogger.log_event("milestone", "MODEL", "load", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "export", milestone=QEFFLogger.MILESTONE_EXPORT_SKIPPED)
    QEFFLogger.log_event("milestone", "MODEL", "compile", milestone=QEFFLogger.MILESTONE_COMPILE_SKIPPED)
    log_api_arguments("generate", "CachedModel", {"max_new_tokens": 1})
    QEFFLogger.log_event("milestone", "MODEL", "generate", milestone=QEFFLogger.MILESTONE_GENERATION_COMPLETE)

    table = QEFFLogger._build_timing_table()
    assert table is not None
    assert "Model Exporting" in table and "0.000" in table
    assert "Model Compilation" in table and "0.000" in table


def test_pipeline_api_arguments_and_nested_records_are_simplified(tmp_path):
    class TestPipeline:
        @classmethod
        @log_pipeline_api("from_pretrained", "Model loading completed.", "load_complete")
        def from_pretrained(cls, model_name):
            log_api_arguments("from_pretrained", "ChildModel", {"model_name": model_name})
            return cls()

        @log_pipeline_api("export", "ONNX export completed.", "export_complete")
        def export(self):
            log_api_arguments("export", "ChildModel", {"duplicate": True})

        @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
        def compile(self):
            log_api_arguments("compile", "ChildModel", {"duplicate": True})

    QEFFLogger.get_logger("TestPipeline", "INFO", str(tmp_path))
    pipeline = TestPipeline.from_pretrained("test/model")
    pipeline.export()
    pipeline.compile()
    QEFFLogger.close_logger()

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is None
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines() if line.startswith("{")]
    assert [record["api"] for record in records if record.get("event") == "api_call"] == [
        "from_pretrained",
        "export",
        "compile",
    ]


def test_failure_and_model_lifecycle_cleanup(tmp_path):
    class TinyModel(QEFFBaseModel):
        def get_model_config(self):
            return {}

        def export(self, export_dir=None):
            return Path("tiny.onnx")

        def compile(self, *args, **kwargs):
            return Path("tiny.qpc")

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            return object.__new__(cls)

    class FailingModel:
        @classmethod
        def from_pretrained(cls, model_name):
            raise RuntimeError("load failed")

    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    TinyModel.from_pretrained("tiny-model")
    assert QEFFLogger.has_active_run() is True
    QEFFLogger.finish_run()
    assert QEFFLogger.has_active_run() is False

    wrapped = logging_utils.log_from_pretrained_call(FailingModel.from_pretrained.__func__)
    FailingModel.from_pretrained = classmethod(wrapped)
    with pytest.raises(RuntimeError, match="load failed"):
        FailingModel.from_pretrained("bad-model")
    assert QEFFLogger.has_active_run() is False

    QEFFLogger.start_run("first")
    QEFFLogger.log_event("milestone", "MODEL", "loaded", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
    QEFFLogger.start_run("second")
    QEFFLogger.finish_run()
    QEFFLogger.close_logger()

    contents = "\n".join(path.read_text() for path in tmp_path.glob("*.log"))
    assert '"api": "from_pretrained"' in contents
    assert "===== QEfficient Timing Summary: tiny-model =====" in contents
    assert "===== QEfficient Timing Summary: first =====" in contents
    assert "===== QEfficient Timing Summary: second =====" in contents


def test_runs_are_independent_across_threads(tmp_path):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    errors = []

    def load_model(name):
        try:
            QEFFLogger.start_run(name)
            QEFFLogger.log_event("milestone", "MODEL", "loaded", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
            QEFFLogger.log_event("milestone", "MODEL", "exported", milestone=QEFFLogger.MILESTONE_EXPORT_COMPLETE)
            QEFFLogger.finish_run()
        except Exception as exc:  # instrumentation must never escape
            errors.append(exc)

    threads = [threading.Thread(target=load_model, args=(f"model-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert QEFFLogger.has_active_run() is False
    assert "===== QEfficient Timing Summary: model-" in next(tmp_path.glob("*.log")).read_text()


def test_configuration_and_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        logging_utils,
        "LoggerConfig",
        replace(LoggerConfig(), max_bytes=256, backup_count=3),
    )
    requested = tmp_path / "shared.log"
    QEFFLogger.get_logger("INFRA", "INFO", str(requested))
    assert QEFFLogger.get_logfile_path() == str(tmp_path / f"shared_{os.getpid()}.log")
    QEFFLogger.set_loglevel("DEBUG")
    QEFFLogger.get_logger("INFRA").debug("visible debug")
    QEFFLogger.start_run("rotating-model")
    QEFFLogger.log_event("milestone", "MODEL", "load", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "export", milestone=QEFFLogger.MILESTONE_EXPORT_COMPLETE)
    QEFFLogger.log_event("milestone", "MODEL", "compile", milestone=QEFFLogger.MILESTONE_COMPILE_COMPLETE)
    QEFFLogger.get_logger("MODEL").info("x" * 2000)
    QEFFLogger.get_logger("INFRA").debug("visible debug after rotation")
    assert QEFFLogger._build_timing_table() is not None
    QEFFLogger.finish_run()
    QEFFLogger.close_logger()
    contents = "\n".join(path.read_text() for path in tmp_path.glob("*.log"))
    assert "visible debug after rotation" in contents
    assert "===== QEfficient Timing Summary: rotating-model =====" in contents
