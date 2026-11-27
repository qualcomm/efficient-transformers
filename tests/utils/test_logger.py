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

TIMING_MILESTONES = {
    name: name
    for name in (
        "load_start",
        "load_complete",
        "export_start",
        "export_complete",
        "export_skipped",
        "compile_start",
        "compile_complete",
        "compile_skipped",
        "mdp_generation_start",
        "mdp_generation_complete",
        "generation_start",
        "generation_complete",
    )
}


def _timing_rows(table):
    assert table is not None
    return {
        cells[1].strip(): cells[2].strip()
        for line in table.splitlines()
        if line.startswith("|") and (cells := line.split("|")) and len(cells) >= 3
    }


def _timing_rows_for(*events):
    state = RunState(run_id=1, model="test")
    for event in events:
        record_timing_milestone(state, *event)
    return _timing_rows(build_timing_table(state, TIMING_MILESTONES))


def _log_milestones(*milestones):
    for milestone in milestones:
        QEFFLogger.log_event("milestone", "MODEL", milestone, milestone=milestone)


@pytest.fixture(autouse=True)
def reset_logger_state():
    QEFFLogger.close_logger()
    yield
    QEFFLogger.close_logger()
    # Keep process-global logger initialized for tests importing module-level adapters.
    QEFFLogger.get_logger("INFRA")


def test_structured_logging_and_console_output(tmp_path, capsys, monkeypatch):
    class TestModel:
        @log_generate_call
        def generate(self, generation_len=4):
            return generation_len * 2

    monkeypatch.setenv(LoggerConfig.log_to_console_env, "1")
    logger = QEFFLogger.get_logger("test", "INFO", str(tmp_path))
    QEFFLogger.start_run("test-model")
    logger.info("visible on console")
    logger.warning_once("cache is disabled for %s", "training")
    logger.warning_once("cache is disabled for %s", "training")
    log_api_arguments("compile", "QEFFTestModel", {"token": "secret", "path": Path("/tmp/model")})
    assert TestModel().generate() == 8
    QEFFLogger.finish_run()
    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    QEFFLogger.close_logger()
    console_output = capsys.readouterr().err

    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.startswith("{")]
    api_calls = {record["api"]: record["arguments"] for record in records if record.get("event") == "api_call"}
    assert api_calls["compile"] == {"token": "<redacted>", "path": "/tmp/model"}
    assert api_calls["generate"] == {"generation_len": 4}
    milestones = {record.get("milestone") for record in records if record.get("event") == "milestone"}
    assert QEFFLogger.MILESTONE_GENERATION_START in milestones
    assert "Generation completed." in {record["message"] for record in records}
    assert sum(record["message"] == "cache is disabled for training" for record in records) == 1
    assert console_output.count("cache is disabled for training") == 1
    assert "visible on console" in console_output
    assert "===== QEfficient Timing Summary: test-model =====" in console_output
    assert "===== QEfficient Timing Summary: test-model =====" in Path(log_path).read_text()


def test_timing_table_scenarios(tmp_path, capsys):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    QEFFLogger.start_run("test-model")
    _log_milestones(
        QEFFLogger.MILESTONE_LOAD_COMPLETE,
        QEFFLogger.MILESTONE_EXPORT_COMPLETE,
        QEFFLogger.MILESTONE_COMPILE_COMPLETE,
        QEFFLogger.MILESTONE_GENERATION_COMPLETE,
    )

    assert QEFFLogger.print_table() is True
    output = capsys.readouterr().out
    assert "Model Loading" in output
    assert "Model Exporting" in output
    assert "Model Compilation" in output
    assert "MDP Generation" not in output
    assert "Text Generation" in output
    assert "Total Time" in output

    QEFFLogger.start_run("cached-model")
    _log_milestones(
        QEFFLogger.MILESTONE_LOAD_COMPLETE,
        QEFFLogger.MILESTONE_EXPORT_SKIPPED,
        QEFFLogger.MILESTONE_COMPILE_SKIPPED,
    )
    log_api_arguments("generate", "CachedModel", {"max_new_tokens": 1})
    _log_milestones(QEFFLogger.MILESTONE_GENERATION_COMPLETE)
    cached_rows = _timing_rows(QEFFLogger._build_timing_table())
    assert cached_rows["Model Exporting"] == "0.000"
    assert cached_rows["Model Compilation"] == "0.000"

    # A cached module must not erase another module's overlapping real work.
    mixed_rows = _timing_rows_for(
        ("load_start", 0.0, "MODEL"),
        ("load_complete", 0.874, "MODEL"),
        ("export_start", 4.658, "CausalLM"),
        ("export_start", 4.804, "Vision"),
        ("export_skipped", 4.850, "Vision"),
        ("export_complete", 607.145, "CausalLM"),
        ("compile_start", 607.425, "CausalLM"),
        ("compile_start", 607.500, "Vision"),
        ("compile_skipped", 607.550, "Vision"),
        ("compile_complete", 652.658, "CausalLM"),
        ("export_start", 652.732, "CausalLM"),
        ("export_skipped", 652.735, "CausalLM"),
        ("compile_start", 653.091, "CausalLM"),
        ("compile_skipped", 653.092, "CausalLM"),
    )
    assert mixed_rows["Model Loading"] == "4.658"
    assert mixed_rows["Model Exporting"] == "602.487"
    assert mixed_rows["Model Compilation"] == "45.233"

    table_rows = _timing_rows_for(
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
    )
    assert table_rows["Model Loading"] == "2.000"
    assert table_rows["Model Exporting"] == "7.000"
    assert table_rows["MDP Generation"] == "2.500"
    assert table_rows["Model Compilation"] == "3.500"
    assert table_rows["Text Generation"] == "4.000"
    assert table_rows["Total Time"] == "19.000"

    table_rows = _timing_rows_for(
        ("load_start", 1.0),
        ("load_complete", 2.0),
        ("export_start", 3.0),
        ("export_complete", 5.0),
        ("compile_start", 6.0),
        ("generation_start", 8.0),
        ("compile_complete", 10.0),
        ("generation_complete", 14.0),
    )
    assert table_rows["Model Compilation"] == "4.000"
    assert table_rows["Text Generation"] == "4.000"
    assert table_rows["Total Time"] == "13.000"


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
    _log_milestones(QEFFLogger.MILESTONE_LOAD_COMPLETE)
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
            _log_milestones(QEFFLogger.MILESTONE_LOAD_COMPLETE, QEFFLogger.MILESTONE_EXPORT_COMPLETE)
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
    _log_milestones(
        QEFFLogger.MILESTONE_LOAD_COMPLETE,
        QEFFLogger.MILESTONE_EXPORT_COMPLETE,
        QEFFLogger.MILESTONE_COMPILE_COMPLETE,
    )
    QEFFLogger.get_logger("MODEL").info("x" * 2000)
    QEFFLogger.get_logger("INFRA").debug("visible debug after rotation")
    assert QEFFLogger._build_timing_table() is not None
    QEFFLogger.finish_run()
    QEFFLogger.close_logger()
    contents = "\n".join(path.read_text() for path in tmp_path.glob("*.log"))
    assert "visible debug after rotation" in contents
    assert "===== QEfficient Timing Summary: rotating-model =====" in contents
