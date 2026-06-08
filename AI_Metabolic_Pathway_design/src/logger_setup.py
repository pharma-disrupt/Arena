"""
Logger Setup Module

Provides pipeline-aware logging with stage tracking, JSON contract logging,
and structured error reporting. Writes to both console and file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from pipeline_config import LoggingConfig, PipelineConfig


# ──────────────────────────────────────────────────────────────────────────────
# STAGE-AWARE LOG FORMATTER
# ──────────────────────────────────────────────────────────────────────────────

class StageFormatter(logging.Formatter):
    """Custom formatter that injects stage information."""

    def __init__(self, fmt: str, datefmt: str, stage: str = "UNKNOWN") -> None:
        super().__init__(fmt, datefmt)
        self._stage = stage

    def format(self, record: logging.LogRecord) -> str:
        record.stage = self._stage  # type: ignore[attr-defined]
        return super().format(record)

    @property
    def stage(self) -> str:
        return self._stage

    @stage.setter
    def stage(self, value: str) -> None:
        self._stage = value


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class PipelineLogger:
    """
    Stage-aware logger that writes to console and rotating log file.

    Usage:
        pl = PipelineLogger(config)
        pl.set_stage("1")
        pl.info("Stage 1 starting...")
    """

    _instance: Optional["PipelineLogger"] = None

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config: PipelineConfig = config or PipelineConfig()
        self._current_stage: str = "UNKNOWN"
        self._log_dir: str = self.config.logging.log_dir
        self._timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file: str = os.path.join(
            self._log_dir,
            f"{self.config.logging.log_file_prefix}_{self._timestamp}.log",
        )

        os.makedirs(self._log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"pipeline_{self._timestamp}")
        if self.logger.handlers:
            self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.logging.console_level)
        console_formatter = StageFormatter(
            self.config.logging.log_format,
            self.config.logging.date_format,
            stage=self._current_stage,
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self._console_handler = console_handler
        self._console_formatter = console_formatter

        # File handler (rotating)
        file_handler = RotatingFileHandler(
            self._log_file,
            maxBytes=self.config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.logging.backup_count,
        )
        file_handler.setLevel(self.config.logging.file_level)
        file_formatter = StageFormatter(
            self.config.logging.log_format,
            self.config.logging.date_format,
            stage=self._current_stage,
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
        self._file_formatter = file_formatter

        PipelineLogger._instance = self
        self.logger.info("PipelineLogger initialised. Log file: %s", self._log_file)

    @classmethod
    def get_instance(cls) -> "PipelineLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_stage(self, stage: str) -> None:
        """Update the stage identifier for all formatters."""
        self._current_stage = stage
        self._console_formatter.stage = stage  # type: ignore[attr-defined]
        self._file_formatter.stage = stage  # type: ignore[attr-defined]

    # ── Convenience wrappers ─────────────────────────────────────────────

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

    def log_json_payload(self, payload: Dict[str, Any], direction: str = "output") -> None:
        """Pretty-print a JSON contract payload at DEBUG level."""
        try:
            summary = _summarise_json(payload, max_depth=2)
            self.info("JSON contract [%s]: %s", direction, summary)
            self.debug("Full JSON payload:\\n%s", json.dumps(payload, indent=2))
        except Exception as exc:
            self.error("Failed to serialise JSON payload: %s", exc)

    def log_error_with_context(
        self,
        function_name: str,
        exc: Exception,
        input_json: Optional[Dict[str, Any]] = None,
        fallback_method: Optional[str] = None,
    ) -> None:
        """Log an error with full context including input JSON and traceback."""
        tb = traceback.format_exc()
        self.error("Exception in %s: %s: %s", function_name, type(exc).__name__, exc)
        if input_json is not None:
            self.error("Input JSON that caused error: %s",
                       json.dumps(input_json, indent=2)[:4000])
        self.error("Traceback:\\n%s", tb)
        if fallback_method:
            self.warning("Attempting fallback: %s", fallback_method)

    def get_log_file_path(self) -> str:
        return self._log_file

    def write_stage_summary(self, stage_number: int, summary: Dict[str, Any]) -> None:
        """Write a stage summary JSON to the logs directory."""
        summary_path = os.path.join(self._log_dir, f"stage_{stage_number}_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)
            self.info("Stage %d summary written to %s", stage_number, summary_path)
        except Exception as exc:
            self.error("Failed to write stage summary: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# ERROR SUMMARY DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorSummary:
    """Structured error report for a single failure event."""

    stage: str
    function: str
    exception_type: str
    message: str
    traceback: str
    input_json: Optional[Dict[str, Any]] = field(default=None)
    fallback_attempted: bool = False
    fallback_method: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "function": self.function,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback,
            "input_json": self.input_json,
            "fallback_attempted": self.fallback_attempted,
            "fallback_method": self.fallback_method,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# SETUP LOGGER (module-level convenience)
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(config: Optional[PipelineConfig] = None) -> PipelineLogger:
    """Create and return a PipelineLogger instance."""
    return PipelineLogger(config)


# ──────────────────────────────────────────────────────────────────────────────
# JSON CONTRACT LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def log_json_contract(
    logger: PipelineLogger,
    payload: Dict[str, Any],
    stage_label: str,
    direction: str = "output",
) -> None:
    """
    Validate and pretty-print a JSON contract handoff.

    Parameters
    ----------
    logger : PipelineLogger
        Active pipeline logger instance.
    payload : dict
        The JSON payload to log.
    stage_label : str
        Human-readable label, e.g. "Stage 1 → Stage 2".
    direction : str
        Either "output" or "input".
    """
    try:
        summary = _summarise_json(payload, max_depth=2)
        logger.info("JSON contract [%s | %s]: %s", direction, stage_label, summary)
        logger.debug("Full JSON payload:\\n%s", json.dumps(payload, indent=2, default=str))
    except Exception as exc:
        logger.error("Failed to log JSON contract: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _summarise_json(payload: Dict[str, Any], max_depth: int = 2) -> str:
    """Return a short human-readable summary of a JSON dict."""
    parts: List[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            nested_keys = list(value.keys())
            parts.append(f"{key}={{{', '.join(nested_keys[:5])}}}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)} items]")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PipelineConfig(organism_key="ecoli", molecule_key="lycopene")
    pl = PipelineLogger(cfg)
    pl.set_stage("1")
    pl.info("=== STAGE 1 LOGGER TEST ===")
    pl.debug("Debug message — this appears only in file if console_level > DEBUG")
    pl.info("Info message — should appear on console and file")
    pl.warning("Warning message")

    # Test JSON contract logging
    test_payload = {
        "pipeline_id": "test-uuid-123",
        "organism": {"name": "E. coli", "strain": "K-12 MG1655"},
        "stage_1_status": "PASS",
    }
    log_json_contract(pl, test_payload, "Test → Logger", direction="output")

    # Test error logging
    try:
        _ = 1 / 0  # deliberate error
    except Exception as exc:
        pl.log_error_with_context("test_main", exc, input_json=test_payload, fallback_method="rule_based")

    # Test stage summary
    pl.write_stage_summary(1, {"status": "PASS", "test": True})
    print(f"\\nLogger test complete. Log file: {pl.get_log_file_path()}")
