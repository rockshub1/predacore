import pytest
"""
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)

Unit tests for the EGM Audit Logger implementations.
"""
import json
import logging
from pathlib import Path

import pytest
from jarvis._vendor.common.protos import egm_pb2  # For LogEventRequest

from jarvis._vendor.ethical_governance_module.audit_logger import FileAuditLogger

# Use pytest-asyncio for async functions
pytestmark = pytest.mark.asyncio

@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Provides a temporary file path for logging."""
    # tmp_path is a pytest fixture providing a temporary directory unique to the test invocation
    log_dir = tmp_path / "logs"
    # Don't create dir here, let the logger handle it
    # log_dir.mkdir()
    return log_dir / "test_audit.log"

@pytest.fixture
def logger_instance(temp_log_file: Path) -> FileAuditLogger:
    """Provides a FileAuditLogger instance using the temporary file."""
    # Disable default logging propagation for tests if desired
    test_logger = logging.getLogger("TestAuditLogger")
    test_logger.propagate = False
    # Ensure the logger uses the temp path provided by the fixture
    return FileAuditLogger(log_file_path=str(temp_log_file), logger=test_logger)

async def test_log_event_creates_file_and_dir(logger_instance: FileAuditLogger, temp_log_file: Path):
    """Test that logging an event creates the log file and directory if they don't exist."""
    log_entry = egm_pb2.LogEventRequest(
        event_type="TEST_EVENT",
        component="TestComponent"
    )
    await logger_instance.log(log_entry)
    assert temp_log_file.parent.is_dir()
    assert temp_log_file.is_file()

async def test_log_event_writes_json_line(logger_instance: FileAuditLogger, temp_log_file: Path):
    """Test that a logged event is written as a valid JSON line."""
    details_struct = egm_pb2.google_dot_protobuf_dot_struct__pb2.Struct()
    details_struct.update({"key": "value", "number": 123})

    compliance_status = egm_pb2.ComplianceCheckResultMessage(
        is_compliant=True,
        warnings=["Warning 1"],
        justification="Test justification"
    )

    log_entry = egm_pb2.LogEventRequest(
        event_type="COMPLEX_EVENT",
        component="LoggerTest",
        details=details_struct,
        compliance_status=compliance_status
    )
    await logger_instance.log(log_entry)

    # Read the file and check content
    assert temp_log_file.is_file()
    content = temp_log_file.read_text(encoding="utf-8")
    assert content.strip() # Ensure content is not empty
    try:
        logged_data = json.loads(content)
    except json.JSONDecodeError:
        pytest.fail(f"Log file content is not valid JSON: {content}")

    # Check structure and values
    assert "timestamp" in logged_data
    assert logged_data["event_type"] == "COMPLEX_EVENT"
    assert logged_data["component"] == "LoggerTest"
    assert logged_data["details"] == {"key": "value", "number": 123}
    assert logged_data["compliance_status"] is not None
    assert logged_data["compliance_status"]["is_compliant"] is True
    assert logged_data["compliance_status"]["violations"] == [] # Ensure empty list is handled
    assert logged_data["compliance_status"]["warnings"] == ["Warning 1"]
    assert logged_data["compliance_status"]["justification"] == "Test justification"

async def test_log_multiple_events_append(logger_instance: FileAuditLogger, temp_log_file: Path):
    """Test that multiple log events are appended as separate lines."""
    entry1 = egm_pb2.LogEventRequest(event_type="EVENT_1", component="C1")
    entry2 = egm_pb2.LogEventRequest(event_type="EVENT_2", component="C2", details={"info": "more"})

    await logger_instance.log(entry1)
    await logger_instance.log(entry2)

    lines = temp_log_file.read_text(encoding="utf-8").strip().split('\n')
    assert len(lines) == 2

    log1 = json.loads(lines[0])
    log2 = json.loads(lines[1])

    assert log1["event_type"] == "EVENT_1"
    assert log2["event_type"] == "EVENT_2"
    assert log2["details"] == {"info": "more"}

# TODO: Test error handling (e.g., file permissions - harder in unit test, maybe integration)
# TODO: Test log directory creation failure handling (if implemented)
