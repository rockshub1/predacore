"""
Implementation of Audit Logging for the EGM.
"""


class AbstractAuditLogger:
    async def log(self, entry):
        raise NotImplementedError


import asyncio
import datetime
import json
import logging
from pathlib import Path

from predacore._vendor.common.protos import egm_pb2

# Define default log file path relative to project root or configurable
DEFAULT_LOG_FILE = "logs/egm_audit.log"


class FileAuditLogger:
    """
    A simple audit logger that writes structured logs to a file.
    """

    def __init__(
        self,
        log_file_path: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.log_file_path_str = log_file_path or DEFAULT_LOG_FILE
        self.log_file_path = Path(self.log_file_path_str)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()  # Use asyncio lock for async file writes
        self._ensure_log_dir()
        self.logger.info(
            f"FileAuditLogger initialized. Logging to: {self.log_file_path}"
        )

    def _ensure_log_dir(self):
        """Ensures the directory for the log file exists."""
        try:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(
                f"Failed to create log directory {self.log_file_path.parent}: {e}",
                exc_info=True,
            )

    async def log(self, entry: egm_pb2.LogEventRequest):
        """
        Formats and appends a log entry to the audit file asynchronously.

        Args:
            entry: The LogEventRequest protobuf message.
        """
        try:
            log_record = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "event_type": entry.event_type,
                "component": entry.component,
                "details": dict(entry.details)
                if entry.details
                else {},  # Convert Struct to dict
                "compliance_status": None,
            }
            if entry.HasField("compliance_status"):
                log_record["compliance_status"] = {
                    "is_compliant": entry.compliance_status.is_compliant,
                    "violations": [
                        dict(v) for v in entry.compliance_status.violations
                    ],
                    "warnings": list(entry.compliance_status.warnings),
                    "justification": entry.compliance_status.justification,
                }

            log_line = json.dumps(log_record)

            async with self._lock:
                try:
                    self._ensure_log_dir()
                    with open(self.log_file_path, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")
                except OSError as e:
                    self.logger.error(
                        f"Failed to write to audit log file {self.log_file_path}: {e}",
                        exc_info=True,
                    )

        except Exception as e:
            self.logger.error(
                f"Unexpected error during audit logging: {e}", exc_info=True
            )
