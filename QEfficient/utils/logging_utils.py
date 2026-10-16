# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import atexit
import inspect
import json
import logging
import os
import reprlib
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from itertools import islice
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from QEfficient.utils.logging_timing import RunState, build_timing_table

# Import centralized config


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for QEFFLogger."""

    log_path_env: str = "QEFF_LOG_PATH"
    log_level_env: str = "QEFF_LOG_LEVEL"
    default_log_dir: str = os.path.expanduser("~/.cache/qefficient_logs")
    default_level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 10
    milestone_load_start: str = "load_start"
    milestone_load_complete: str = "load_complete"
    milestone_export_complete: str = "export_complete"
    milestone_export_skipped: str = "export_skipped"
    milestone_compile_complete: str = "compile_complete"
    milestone_compile_skipped: str = "compile_skipped"
    milestone_generation_complete: str = "generation_complete"


class JSONNamespaceFormatter(logging.Formatter):
    """
    Custom formatter to output log records in JSON format with metadata.
    """

    def format(self, record):
        log_record = {
            "created": record.created,
            "date": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d"),
            "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "namespace": getattr(record, "namespace", "default"),
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        for field in ("event", "api", "arguments"):
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value
        for field in ("run_id", "model", "milestone", "status", "error"):
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value
        return json.dumps(log_record, default=str)


class QEFFLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with a Transformers-compatible one-time warning helper."""

    _warning_once_lock = threading.Lock()
    _warning_once_keys = set()

    def warning_once(self, msg, *args, **kwargs) -> None:
        """Log a warning once per namespace and rendered message per process."""
        try:
            rendered_message = msg % args if args else str(msg)
        except Exception:
            rendered_message = str(msg)
        key = (self.logger.name, self.extra.get("namespace"), rendered_message)
        with self._warning_once_lock:
            if key in self._warning_once_keys:
                return
            self._warning_once_keys.add(key)
        self.warning(msg, *args, **kwargs)


_SENSITIVE_ARGUMENT_NAMES = {
    "api_key",
    "authorization",
    "access_token",
    "hf_token",
    "password",
    "secret",
    "token",
    "use_auth_token",
}
_MAX_SERIALIZED_ITEMS = 64
_MAX_SERIALIZED_DEPTH = 3


def _serialize_argument(value: Any, name: Optional[str] = None, *, _depth: int = 0) -> Any:
    """Return a compact JSON-safe representation of an API argument."""
    if name and name.lower() in _SENSITIVE_ARGUMENT_NAMES:
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth >= _MAX_SERIALIZED_DEPTH:
        return {"type": type(value).__name__, "repr": "<nested value omitted>"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        items = list(islice(value.items(), _MAX_SERIALIZED_ITEMS))
        result = {str(key): _serialize_argument(item, str(key), _depth=_depth + 1) for key, item in items}
        if len(value) > len(items):
            result["__truncated__"] = len(value) - len(items)
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(islice(iter(value), _MAX_SERIALIZED_ITEMS))
        result = [_serialize_argument(item, _depth=_depth + 1) for item in items]
        if len(value) > len(items):
            result.append(f"<truncated {len(value) - len(items)} items>")
        return result
    try:
        shape = getattr(value, "shape")
        dtype = getattr(value, "dtype")
        result = {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
        try:
            result["device"] = str(getattr(value, "device"))
        except Exception:
            pass
        return result
    except Exception:
        # Some tensor-like or lazy objects expose properties that can raise
        # arbitrary exceptions. Logging must not interrupt the API call.
        pass
    try:
        to_dict = getattr(value, "to_dict")
        if callable(to_dict):
            config = to_dict()
            return {
                "type": f"{type(value).__module__}.{type(value).__name__}",
                "config_keys": [str(key) for key in islice(config, _MAX_SERIALIZED_ITEMS)],
                "config_key_count": len(config),
            }
    except Exception:
        pass
    try:
        safe_repr = reprlib.repr(value)
    except Exception:
        safe_repr = "<unrepresentable value>"
    return {
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": safe_repr,
    }


def log_api_arguments(api: str, namespace: str, arguments: Dict[str, Any]) -> None:
    """Write one structured API-argument record."""
    QEFFLogger.log_event(
        "api_call",
        namespace,
        "Captured API arguments.",
        api=api,
        arguments={key: _serialize_argument(value, key) for key, value in arguments.items()},
    )


def log_generate_call(func):
    """Log generate arguments and a plain completion message."""
    signature = inspect.signature(func)
    receiver_name = next(iter(signature.parameters), None)

    @wraps(func)
    def wrapper(*args, **kwargs):
        receiver = args[0]
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop(receiver_name, None)
        namespace = type(receiver).__name__
        log_api_arguments("generate", namespace, arguments)
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            QEFFLogger.log_api_failure("generate", namespace, exc)
            raise
        QEFFLogger.log_event(
            "milestone",
            namespace,
            "Generation completed.",
            api="generate",
            milestone=QEFFLogger.MILESTONE_GENERATION_COMPLETE,
        )
        return result

    wrapper._qeff_api_dump = True
    return wrapper


def log_pipeline_api(
    api: str,
    completion_message: str,
    completion_milestone: str,
    start_run: bool = False,
    finish_run: bool = False,
):
    """Log one public Diffusers API and optionally manage its timing run.

    ``from_pretrained``, ``export``, and ``compile`` suppress duplicate API
    records emitted by child models.  A pipeline ``__call__`` can use the same
    decorator with ``start_run=True`` and ``finish_run=True`` to log generate
    arguments and close the timing run.
    """

    def decorator(func):
        signature = inspect.signature(func)
        receiver_name = next(iter(signature.parameters), None)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            receiver = arguments.pop(receiver_name, None)
            namespace = receiver.__name__ if isinstance(receiver, type) else type(receiver).__name__
            if api == "from_pretrained":
                QEFFLogger.start_run(str(arguments.get("pretrained_model_name_or_path", "unknown")))
            elif start_run and not QEFFLogger.has_active_run():
                QEFFLogger.start_run(namespace)

            log_api_arguments(api, namespace, arguments)
            try:
                if api in {"from_pretrained", "export", "compile"}:
                    with QEFFLogger.suppress_api_logging():
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception as exc:
                QEFFLogger.log_api_failure(api, namespace, exc)
                if api == "from_pretrained":
                    QEFFLogger.finish_run()
                raise
            else:
                QEFFLogger.log_event(
                    "milestone",
                    namespace,
                    completion_message,
                    api=api,
                    milestone=completion_milestone,
                )
                return result
            finally:
                if finish_run:
                    QEFFLogger.finish_run()

        wrapper._qeff_api_dump = True
        return wrapper

    return decorator


def log_from_pretrained_call(func):
    """Start a model run and capture ``from_pretrained`` arguments before loading."""
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        bound = signature.bind_partial(cls, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("cls", None)
        model_name = arguments.get("pretrained_model_name_or_path", "unknown")
        if QEFFLogger.is_api_logging_suppressed():
            return func(cls, *args, **kwargs)
        QEFFLogger.start_run(str(model_name))
        log_api_arguments("from_pretrained", cls.__name__, arguments)
        try:
            result = func(cls, *args, **kwargs)
        except Exception as exc:
            QEFFLogger.log_api_failure("from_pretrained", cls.__name__, exc)
            QEFFLogger.finish_run()
            raise
        else:
            QEFFLogger.log_event(
                "milestone",
                cls.__name__,
                "Model loading completed.",
                api="from_pretrained",
                milestone=QEFFLogger.MILESTONE_LOAD_COMPLETE,
            )
            return result

    wrapper._qeff_api_dump = True
    return wrapper


class QEFFLogger:
    """
    Singleton logger class for structured logging with namespace support.

    Project-wide behavior:
      - A single log level is enforced using env `QEFF_LOG_LEVEL` (default = INFO).
      - Log path resolved with priority: explicit arg > env `QEFF_LOG_PATH` > default dir + timestamp.
    """

    _instance: Optional[logging.Logger] = None
    _logfile: Optional[str] = None
    _init_lock = threading.Lock()
    _summary_printed = False
    _atexit_registered = False
    _print_logfile_at_exit = True
    _fallback_warning_emitted = False
    _run_counter = 0
    _runs: Dict[int, RunState] = {}
    _run_local = threading.local()
    _api_logging_local = threading.local()
    _run_lock = threading.RLock()

    MILESTONE_LOAD_START = LoggerConfig.milestone_load_start
    MILESTONE_LOAD_COMPLETE = LoggerConfig.milestone_load_complete
    MILESTONE_EXPORT_COMPLETE = LoggerConfig.milestone_export_complete
    MILESTONE_EXPORT_SKIPPED = LoggerConfig.milestone_export_skipped
    MILESTONE_COMPILE_COMPLETE = LoggerConfig.milestone_compile_complete
    MILESTONE_COMPILE_SKIPPED = LoggerConfig.milestone_compile_skipped
    MILESTONE_GENERATION_COMPLETE = LoggerConfig.milestone_generation_complete

    def __init__(self, loglevel: Optional[str] = None, log_path: Optional[str] = None):
        """
        Initialize the logger instance with specified path. Level is globally controlled by env.
        Args:
            loglevel: kept for backward compatibility, but env `QEFF_LOG_LEVEL` takes precedence.
            log_path: optional path to the log file (highest priority).
        """
        with QEFFLogger._init_lock:
            if QEFFLogger._instance is not None:
                return

            # Determine effective log level:
            # Priority: ENV(QEFF_LOG_LEVEL) -> arg(loglevel) -> LoggerConfig.default_level
            env_level = os.environ.get(LoggerConfig.log_level_env)
            effective_level_name = (env_level or loglevel or LoggerConfig.default_level).upper()
            numeric_level = getattr(logging, effective_level_name, None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {effective_level_name}")
            self.loglevel = effective_level_name

            # Resolve log path (arg > env > default dir + timestamp). Logging
            # must never make importing or using QEfficient fail.
            env_path = os.environ.get(LoggerConfig.log_path_env)
            try:
                self.log_path = self._resolve_log_path(log_path or env_path)
            except Exception as exc:
                self.log_path = None
                self._warn_file_logging_fallback(exc)

            # Initialize the base logger
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger
            QEFFLogger._summary_printed = False
            if not QEFFLogger._atexit_registered:
                atexit.register(QEFFLogger._finalize)
                QEFFLogger._atexit_registered = True

    @classmethod
    def _resolve_log_path(cls, requested_path: Optional[str]) -> str:
        """Resolve the final log file path from a user path or defaults."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_suffix = f"_{os.getpid()}"
        default_file = os.path.join(LoggerConfig.default_log_dir, f"QEFF_{timestamp}{process_suffix}.log")
        if not requested_path:
            os.makedirs(LoggerConfig.default_log_dir, exist_ok=True)
            return default_file

        path = Path(requested_path).expanduser()
        if path.suffix.lower() == ".log":
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path.with_name(f"{path.stem}_{os.getpid()}{path.suffix}"))

        path.mkdir(parents=True, exist_ok=True)
        return str(path / f"QEFF_{timestamp}{process_suffix}.log")

    @classmethod
    def _warn_file_logging_fallback(cls, error: Exception) -> None:
        """Warn once when file logging is unavailable."""
        if cls._fallback_warning_emitted:
            return
        cls._fallback_warning_emitted = True
        warnings.warn(
            f"QEfficient file logging is unavailable; using console logging instead: {error}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _initialize_logger(self) -> logging.Logger:
        """
        Set up the logger with rotating file handler and JSON formatter.
        """
        QEFFLogger._logfile = self.log_path

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(getattr(logging, self.loglevel))
        logger.propagate = False

        # Avoid duplicate handlers if reinitialized in same process
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        try:
            if self.log_path is None:
                raise OSError("log path resolution failed")
            handler: logging.Handler = RotatingFileHandler(
                self.log_path,
                maxBytes=LoggerConfig.max_bytes,
                backupCount=LoggerConfig.backup_count,
                delay=True,
            )
        except Exception as exc:
            QEFFLogger._logfile = None
            self._warn_file_logging_fallback(exc)
            handler = logging.StreamHandler()

        class RunContextFilter(logging.Filter):
            def filter(self, record):
                state = QEFFLogger._get_run_state()
                if state is not None:
                    record.run_id = state.run_id
                    record.model = state.model
                return True

        handler.addFilter(RunContextFilter())
        handler.setFormatter(JSONNamespaceFormatter())
        logger.addHandler(handler)

        return logger

    @classmethod
    def get_logger(
        cls, namespace: str, loglevel: Optional[str] = None, log_path: Optional[str] = None
    ) -> logging.Logger:
        """
        Retrieve a logger adapter with a specific namespace.
        Note: project-wide level comes from env `QEFF_LOG_LEVEL` (default INFO).
        """
        if cls._instance is None:
            cls(loglevel, log_path)
        return QEFFLoggerAdapter(cls._instance, {"namespace": namespace})

    @classmethod
    def log_event(cls, event: str, namespace: str, message: str, **fields: Any) -> None:
        """Write one structured event record."""
        with cls._run_lock:
            if cls._is_api_logging_suppressed() and event in {"api_call", "milestone", "api_failure"}:
                return
            if cls._instance is None:
                cls.get_logger(namespace)
            state = cls._get_run_state()
            if state is not None:
                now = datetime.now().timestamp()
                milestone = fields.get("milestone")
                if event == "milestone" and milestone:
                    if milestone == cls.MILESTONE_LOAD_START:
                        state.milestones.setdefault(milestone, now)
                    else:
                        state.milestones[milestone] = now
                if event == "api_call" and fields.get("api") == "generate":
                    state.milestones.setdefault("generate_start", now)
                if event == "api_failure" and fields.get("api") == "compile":
                    state.milestones["compile_failed"] = now
            extra = {"namespace": namespace, "event": event, **fields}
            logging.LoggerAdapter(cls._instance, extra).info(message, stacklevel=2)

    @classmethod
    def _get_run_state(cls) -> Optional[RunState]:
        with cls._run_lock:
            run_id = getattr(cls._run_local, "run_id", None)
            return cls._runs.get(run_id) if run_id is not None else None

    @classmethod
    def has_active_run(cls) -> bool:
        return cls._get_run_state() is not None

    @classmethod
    def log_api_failure(cls, api: str, namespace: str, error: Exception) -> None:
        cls.log_event(
            "api_failure",
            namespace,
            f"{api} failed.",
            api=api,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )

    @classmethod
    @contextmanager
    def suppress_api_logging(cls):
        depth = getattr(cls._api_logging_local, "depth", 0)
        cls._api_logging_local.depth = depth + 1
        try:
            yield
        finally:
            cls._api_logging_local.depth = max(0, getattr(cls._api_logging_local, "depth", 1) - 1)

    @classmethod
    def is_api_logging_suppressed(cls) -> bool:
        return cls._is_api_logging_suppressed()

    @classmethod
    def _is_api_logging_suppressed(cls) -> bool:
        return getattr(cls._api_logging_local, "depth", 0) > 0

    @classmethod
    def log(cls, level: str, namespace: str, msg: str):
        """
        Log a message with specified level and metadata.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized. Call get_logger() first.")

        level_num = getattr(logging, level.upper(), None)
        if not isinstance(level_num, int):
            raise ValueError(f"Invalid log level: {level}")

        logger = logging.LoggerAdapter(cls._instance, {"namespace": namespace})
        logger.log(level_num, msg, stacklevel=2)

    @classmethod
    def set_loglevel(cls, loglevel: Optional[str] = None):
        """
        Update the log level of the logger at runtime.
        Priority remains ENV > arg > default.
        If ENV is set, it will continue to override; otherwise arg/default apply.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized yet. Call get_logger() first.")

        env_level = os.environ.get(LoggerConfig.log_level_env)
        effective_level_name = (env_level or loglevel or LoggerConfig.default_level).upper()
        numeric_level = getattr(logging, effective_level_name, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {effective_level_name}")

        cls._instance.setLevel(numeric_level)

    @classmethod
    def close_logger(cls):
        """
        Gracefully shut down the logger.
        """
        with cls._run_lock:
            if cls._instance:
                cls._finalize(print_log_path=False)
                cls._close_handlers()
            cls._runs.clear()
            cls._run_local.__dict__.clear()

    @classmethod
    def _close_handlers(cls) -> None:
        if cls._instance is None:
            return
        for handler in cls._instance.handlers[:]:
            handler.flush()
            handler.close()
            cls._instance.removeHandler(handler)
        cls._instance = None
        cls._logfile = None

    @classmethod
    def start_run(cls, model_name: str) -> None:
        """Start an independent model run for the calling thread."""
        with cls._run_lock:
            if cls._instance is None:
                cls()
            previous_state = cls._get_run_state()
            if previous_state is not None:
                # Do not silently orphan a run when callers load another model
                # without explicitly finishing the previous one.
                cls._finish_state(previous_state)
            cls._run_counter += 1
            state = RunState(cls._run_counter, model_name)
            cls._runs[state.run_id] = state
            cls._run_local.run_id = state.run_id
            cls._summary_printed = False
            cls.log_event(
                "milestone",
                "MODEL",
                f"Starting model weight loading: {model_name}.",
                milestone=cls.MILESTONE_LOAD_START,
            )

    @classmethod
    def finish_run(cls) -> None:
        """Best-effort finalization; instrumentation must never affect callers."""
        with cls._run_lock:
            state = cls._get_run_state()
            if state is None:
                return
            cls._finish_state(state)

    @classmethod
    def _finish_state(cls, state: RunState) -> None:
        """Finalize and detach one run, without relying on thread-local context."""
        try:
            cls._append_final_table(state)
        except Exception:
            logging.getLogger(__name__).debug("Unable to write QEfficient timing summary", exc_info=True)
        finally:
            cls._runs.pop(state.run_id, None)
            if getattr(cls._run_local, "run_id", None) == state.run_id:
                cls._run_local.__dict__.clear()

    @classmethod
    def _build_timing_table(cls, run_id: Optional[int] = None) -> Optional[str]:
        """Build a partial timing table from in-memory structured milestones."""
        with cls._run_lock:
            if run_id is None:
                state = cls._get_run_state()
            else:
                state = cls._runs.get(run_id)
            return build_timing_table(
                state,
                {
                    "load_start": cls.MILESTONE_LOAD_START,
                    "load_complete": cls.MILESTONE_LOAD_COMPLETE,
                    "export_complete": cls.MILESTONE_EXPORT_COMPLETE,
                    "export_skipped": cls.MILESTONE_EXPORT_SKIPPED,
                    "compile_complete": cls.MILESTONE_COMPILE_COMPLETE,
                    "compile_skipped": cls.MILESTONE_COMPILE_SKIPPED,
                    "generation_complete": cls.MILESTONE_GENERATION_COMPLETE,
                },
            )

    @classmethod
    def _append_final_table(cls, state: RunState, table: Optional[str] = None) -> bool:
        """Append one human-readable timing summary to the active log file."""
        with cls._run_lock:
            if state.table_written or cls._instance is None:
                return False
            table = table or cls._build_timing_table(state.run_id)
            if table is None:
                return False
            if cls._logfile is None:
                return False
            for handler in cls._instance.handlers:
                handler.flush()
            with open(cls._logfile, "a", encoding="utf-8") as handle:
                handle.write(f"\n===== QEfficient Timing Summary: {state.model} =====\n")
                handle.write(table)
                handle.write("\n\n")
            state.table_written = True
            return True

    @classmethod
    def _finalize(cls, *, print_log_path: bool = True) -> None:
        """Finalize all remaining runs once at process shutdown."""
        with cls._run_lock:
            if cls._instance is None or cls._summary_printed:
                return
            states = list(cls._runs.values())
            for state in states:
                try:
                    cls._append_final_table(state)
                except Exception:
                    logging.getLogger(__name__).debug("Unable to finalize QEfficient logging", exc_info=True)
            cls._runs.clear()
            cls._summary_printed = True
            # This terminal output is intentional: standalone scripts need a
            # discoverable path after completion. Pytest suppresses it for its
            # controller and prints worker paths separately.
            if print_log_path and cls._print_logfile_at_exit and cls._logfile:
                print(f"Log file: {cls._logfile}", flush=True)

    @classmethod
    def get_logfile_path(cls) -> Optional[str]:
        """Return active log file path, if logger is initialized."""
        return cls._logfile

    @classmethod
    def print_table(cls) -> bool:
        """
        Append and print the timing table with t1 as baseline (0.0s).
        """
        table = cls._build_timing_table()
        if table is None:
            return False
        state = cls._get_run_state()
        if state is not None:
            try:
                cls._append_final_table(state, table)
            except Exception:
                logging.getLogger(__name__).debug("Unable to write QEfficient timing summary", exc_info=True)
        print("\n" + table)
        return True
