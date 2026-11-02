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
from QEfficient.utils.logging_timing import RunState, build_timing_table, record_timing_milestone
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
    compile_record = next(
        record for record in records if record.get("event") == "api_call" and record.get("api") == "compile"
    )
    generate_record = next(
        record for record in records if record.get("event") == "api_call" and record.get("api") == "generate"
    )
    assert compile_record["arguments"] == {"token": "<redacted>", "path": "/tmp/model"}
    assert generate_record["arguments"] == {"generation_len": 4}
    assert any(
        record.get("event") == "milestone" and record.get("milestone") == QEFFLogger.MILESTONE_GENERATION_START
        for record in records
    )
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
    assert "MDP Generation" not in output
    assert "Text Generation" in output
    assert "Total Time" in output


def test_mdp_generation_is_reported_separately_from_compile_time():
    state = RunState(run_id=1, model="mdp-model")
    for milestone, timestamp in (
        ("load_start", 1.0),
        ("load_complete", 2.0),
        ("export_start", 3.0),
        ("export_complete", 6.0),
        ("compile_start", 10.0),
        ("mdp_generation_start", 11.0),
        ("mdp_generation_complete", 13.5),
        ("compile_complete", 16.0),
        ("generation_start", 16.0),
        ("generation_complete", 20.0),
    ):
        record_timing_milestone(state, milestone, timestamp)

    table = build_timing_table(
        state,
        {
            "load_start": "load_start",
            "load_complete": "load_complete",
            "export_start": "export_start",
            "export_complete": "export_complete",
            "export_skipped": "export_skipped",
            "compile_start": "compile_start",
            "compile_complete": "compile_complete",
            "compile_skipped": "compile_skipped",
            "mdp_generation_start": "mdp_generation_start",
            "mdp_generation_complete": "mdp_generation_complete",
            "generation_start": "generation_start",
            "generation_complete": "generation_complete",
        },
    )

    assert table is not None
    table_rows = {
        cells[1].strip(): cells[2].strip()
        for line in table.splitlines()
        if line.startswith("|") and (cells := line.split("|")) and len(cells) >= 3
    }
    assert table_rows["Model Loading"] == "2.000"
    assert table_rows["Model Exporting"] == "7.000"
    assert table_rows["MDP Generation"] == "2.500"
    assert table_rows["Model Compilation"] == "3.500"
    assert table_rows["Text Generation"] == "4.000"
    assert table_rows["Total Time"] == "19.000"


def test_compile_completion_after_generation_start_sets_generation_boundary():
    state = RunState(run_id=2, model="compile-order-model")
    for milestone, timestamp in (
        ("load_start", 1.0),
        ("load_complete", 2.0),
        ("export_start", 3.0),
        ("export_complete", 5.0),
        ("compile_start", 6.0),
        ("generation_start", 8.0),
        ("compile_complete", 10.0),
        ("generation_complete", 14.0),
    ):
        record_timing_milestone(state, milestone, timestamp)

    table = build_timing_table(
        state,
        {
            "load_start": "load_start",
            "load_complete": "load_complete",
            "export_start": "export_start",
            "export_complete": "export_complete",
            "export_skipped": "export_skipped",
            "compile_start": "compile_start",
            "compile_complete": "compile_complete",
            "compile_skipped": "compile_skipped",
            "mdp_generation_start": "mdp_generation_start",
            "mdp_generation_complete": "mdp_generation_complete",
            "generation_start": "generation_start",
            "generation_complete": "generation_complete",
        },
    )

    assert table is not None
    table_rows = {
        cells[1].strip(): cells[2].strip()
        for line in table.splitlines()
        if line.startswith("|") and (cells := line.split("|")) and len(cells) >= 3
    }
    assert table_rows["Model Compilation"] == "4.000"
    assert table_rows["Text Generation"] == "4.000"
    assert table_rows["Total Time"] == "13.000"


def test_console_logging_env_mirrors_logs_and_timing_table(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(LoggerConfig.log_to_console_env, "1")
    logger = QEFFLogger.get_logger("MODEL", "INFO", str(tmp_path))
    logger.info("visible on console")
    QEFFLogger.start_run("console-model")
    QEFFLogger.log_event("milestone", "MODEL", "load", milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE)
    QEFFLogger.finish_run()

    output = capsys.readouterr().err
    assert "visible on console" in output
    assert "===== QEfficient Timing Summary: console-model =====" in output

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    assert "===== QEfficient Timing Summary: console-model =====" in Path(log_path).read_text()


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
            QEFFLogger.log_event(
                "milestone",
                "ChildModel",
                "Child export skipped.",
                milestone=QEFFLogger.MILESTONE_EXPORT_SKIPPED,
            )

        @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
        def compile(self):
            log_api_arguments("compile", "ChildModel", {"duplicate": True})
            QEFFLogger.log_event(
                "milestone",
                "ChildModel",
                "Compilation started.",
                milestone=QEFFLogger.MILESTONE_COMPILE_START,
            )
            QEFFLogger.log_event(
                "milestone",
                "ChildModel",
                "MDP generation started.",
                milestone=QEFFLogger.MILESTONE_MDP_GENERATION_START,
            )
            QEFFLogger.log_event(
                "milestone",
                "ChildModel",
                "MDP generation completed.",
                milestone=QEFFLogger.MILESTONE_MDP_GENERATION_COMPLETE,
            )
            QEFFLogger.log_event(
                "milestone",
                "ChildModel",
                "Compilation completed.",
                milestone=QEFFLogger.MILESTONE_COMPILE_COMPLETE,
            )

        @log_pipeline_api("generate", "Generation completed.", "generation_complete")
        def generate(self):
            return "generated"

    QEFFLogger.get_logger("TestPipeline", "INFO", str(tmp_path))
    pipeline = TestPipeline.from_pretrained("test/model")
    pipeline.export()
    pipeline.compile()
    assert pipeline.generate() == "generated"
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
        "generate",
    ]
    stage_milestones = [record.get("milestone") for record in records if record.get("event") == "milestone"]
    assert [
        milestone
        for milestone in stage_milestones
        if milestone
        in {
            QEFFLogger.MILESTONE_EXPORT_START,
            QEFFLogger.MILESTONE_EXPORT_COMPLETE,
            QEFFLogger.MILESTONE_EXPORT_SKIPPED,
        }
    ] == [QEFFLogger.MILESTONE_EXPORT_START, QEFFLogger.MILESTONE_EXPORT_COMPLETE]
    assert QEFFLogger.MILESTONE_COMPILE_START in stage_milestones
    assert QEFFLogger.MILESTONE_MDP_GENERATION_START in stage_milestones
    assert QEFFLogger.MILESTONE_MDP_GENERATION_COMPLETE in stage_milestones
    assert QEFFLogger.MILESTONE_GENERATION_START in stage_milestones
    assert QEFFLogger.MILESTONE_GENERATION_COMPLETE in stage_milestones


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
