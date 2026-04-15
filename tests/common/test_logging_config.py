"""
Tests for src/common/logging_config.py — Structured logging with trace_id.
"""
import json
import logging

from jarvis._vendor.common.logging_config import (
    JSONFormatter,
    PrettyFormatter,
    get_logger,
    setup_logging,
)
from jarvis._vendor.common.metrics import set_trace_id

# ── JSONFormatter Tests ──────────────────────────────────────────────

class TestJSONFormatter:
    def setup_method(self):
        self.formatter = JSONFormatter()

    def _make_record(self, msg: str, level: int = logging.INFO, **kwargs):
        record = logging.LogRecord(
            name="test.logger", level=level, pathname="test.py",
            lineno=42, msg=msg, args=(), exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        return record

    def test_basic_json_output(self):
        record = self._make_record("hello world")
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "ts" in data
        assert "trace_id" in data

    def test_trace_id_included(self):
        set_trace_id("test_trace_abc123")
        try:
            record = self._make_record("traced msg")
            output = self.formatter.format(record)
            data = json.loads(output)
            assert data["trace_id"] == "test_trace_abc123"
        finally:
            set_trace_id("")

    def test_error_includes_file_info(self):
        record = self._make_record("oops", level=logging.ERROR)
        output = self.formatter.format(record)
        data = json.loads(output)
        assert "file" in data
        assert "test.py:42" in data["file"]

    def test_warning_includes_file_info(self):
        record = self._make_record("careful", level=logging.WARNING)
        output = self.formatter.format(record)
        data = json.loads(output)
        assert "file" in data

    def test_info_excludes_file_info(self):
        record = self._make_record("normal log")
        output = self.formatter.format(record)
        data = json.loads(output)
        assert "file" not in data

    def test_extra_fields_included(self):
        record = self._make_record("user action", user_id="shubham", channel="cli")
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["user_id"] == "shubham"
        assert data["channel"] == "cli"

    def test_exception_info(self):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = self._make_record("failed", level=logging.ERROR)
            record.exc_info = sys.exc_info()
            output = self.formatter.format(record)
            data = json.loads(output)
            assert data["error"] == "test error"
            assert data["error_type"] == "ValueError"

    def test_output_is_single_line(self):
        record = self._make_record("multi\nline\nmessage")
        output = self.formatter.format(record)
        # JSON output itself is one line (no embedded newlines in JSON keys)
        parsed = json.loads(output)
        assert parsed["msg"] == "multi\nline\nmessage"


# ── PrettyFormatter Tests ────────────────────────────────────────────

class TestPrettyFormatter:
    def setup_method(self):
        self.formatter = PrettyFormatter()

    def _make_record(self, msg: str, level: int = logging.INFO, **kwargs):
        record = logging.LogRecord(
            name="jarvis.core", level=level, pathname="core.py",
            lineno=10, msg=msg, args=(), exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        return record

    def test_basic_output(self):
        record = self._make_record("Processing message")
        output = self.formatter.format(record)
        assert "Processing message" in output
        assert "▸" in output

    def test_trace_id_in_output(self):
        set_trace_id("pretty_trace_id")
        try:
            record = self._make_record("traced")
            output = self.formatter.format(record)
            assert "trace:pretty_t" in output
        finally:
            set_trace_id("")

    def test_extras_in_output(self):
        record = self._make_record("action", user_id="shubham")
        output = self.formatter.format(record)
        assert "user_id=shubham" in output

    def test_error_color_code(self):
        record = self._make_record("error occurred", level=logging.ERROR)
        output = self.formatter.format(record)
        assert "\033[31m" in output  # Red ANSI


# ── setup_logging Tests ──────────────────────────────────────────────

class TestSetupLogging:
    def test_setup_json_mode(self):
        setup_logging(level="DEBUG", json_mode=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
        # Cleanup
        root.handlers.clear()

    def test_setup_pretty_mode(self):
        setup_logging(level="INFO", json_mode=False)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert isinstance(root.handlers[0].formatter, PrettyFormatter)
        root.handlers.clear()

    def test_setup_with_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="INFO", json_mode=False, log_file=log_file)
        root = logging.getLogger()
        assert len(root.handlers) == 1  # file only (no console to avoid double writes)
        file_handler = root.handlers[0]
        assert isinstance(file_handler.formatter, JSONFormatter)
        root.handlers.clear()

    def test_noisy_loggers_suppressed(self):
        setup_logging()
        for name in ["httpx", "httpcore", "urllib3", "asyncio", "grpc"]:
            assert logging.getLogger(name).level >= logging.WARNING
        logging.getLogger().handlers.clear()


# ── get_logger Tests ─────────────────────────────────────────────────

class TestGetLogger:
    def test_get_named_logger(self):
        logger = get_logger("jarvis.core")
        assert logger.name == "jarvis.core"

    def test_loggers_are_cached(self):
        l1 = get_logger("test.module")
        l2 = get_logger("test.module")
        assert l1 is l2

    def test_different_names_different_loggers(self):
        l1 = get_logger("module.a")
        l2 = get_logger("module.b")
        assert l1 is not l2
