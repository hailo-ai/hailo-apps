# hailo_logger.py
from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any

# ---- module state (singleton-ish) ----
_CONFIGURED = False

# Stable run id for this process (not printed by default)
_RUN_ID = (
    os.getenv("HAILO_RUN_ID")
    or datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
)

# Basic string->level map (kept small & obvious)
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _coerce_level(level: str | int | None) -> int:
    """Coerce a string/int/None into a logging level int."""
    if isinstance(level, int):
        return level
    if level is None:
        return logging.INFO
    return _LEVELS.get(str(level).upper(), logging.INFO)


def get_run_id() -> str:
    """Return the stable run id for this process.

    Not shown in log lines by default, but available for:
      * experiment tracking
      * test logs
      * debugging
    """
    return _RUN_ID


def init_logging(
    *,
    level: str | int | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger exactly once (unless force=True).

    Priority for level:
      1) explicit param
      2) env HAILO_LOG_LEVEL
      3) INFO (default)

    If log_file is provided (or $HAILO_LOG_FILE is set),
    logs will also be written to that file.

    Noisy Logger Suppression:
      Internal loggers (hailo_apps.installation.*) are suppressed to INFO
      when DEBUG is enabled via CLI, but NOT suppressed when DEBUG is set via
      environment variable (HAILO_LOG_LEVEL).

    This is the only place that should touch handlers / root config.
    All other code just calls get_logger(name).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    # Resolve level from param or env
    env_level = os.getenv("HAILO_LOG_LEVEL")
    # Track if level came from environment variable (user's explicit choice)
    level_from_env = env_level is not None and level is None
    resolved_level = _coerce_level(level if level is not None else env_level)

    # Clear existing handlers to avoid duplicates (tests/notebooks/CLI reuse)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(resolved_level)

    # Full format for DEBUG mode (with timestamp, run_id, full name)
    debug_fmt = "%(asctime)s | %(levelname)s | run=%(run_id)s | %(name)s | %(message)s"
    # Concise format for normal mode (severity, short name, message)
    normal_fmt = "%(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    # Console handler with custom formatter
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ShortNameFormatter(debug_fmt=debug_fmt, normal_fmt=normal_fmt, datefmt=datefmt))
    ch.addFilter(_RunContextFilter(_RUN_ID))
    root.addHandler(ch)

    # Optional file handler (always use full format for files)
    log_file = log_file or os.getenv("HAILO_LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt=debug_fmt, datefmt=datefmt))
        fh.addFilter(_RunContextFilter(_RUN_ID))
        root.addHandler(fh)

    # Be quiet about common noisy deps unless user explicitly wants DEBUG
    logging.getLogger("urllib3").setLevel(max(resolved_level, logging.WARNING))
    logging.getLogger("PIL").setLevel(max(resolved_level, logging.WARNING))

    # Suppress noisy internal loggers that aren't relevant for application debugging
    # These loggers will be set to INFO level even when DEBUG is enabled via CLI
    # However, if user explicitly sets DEBUG via environment variable, respect their choice
    # and don't suppress (they want to see everything)
    noisy_loggers = [
        "hailo_apps.installation.config_utils",  # Config loading is verbose
        "hailo_apps.installation",  # Suppress entire installation module
    ]

    # Only suppress noisy loggers if level did NOT come from environment variable
    # If user explicitly set HAILO_LOG_LEVEL=DEBUG, they want to see all logs
    if not level_from_env:
        # Set noisy loggers to INFO (or WARNING if root is INFO/ERROR)
        # This keeps them quiet during DEBUG mode but still shows important messages
        noisy_level = logging.INFO if resolved_level == logging.DEBUG else max(resolved_level, logging.WARNING)
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(noisy_level)

    # Ensure all existing loggers inherit from root logger level
    # This fixes the issue where loggers created before init_logging() might not inherit DEBUG level
    # We iterate through all existing loggers and ensure they propagate to root
    # This is important because loggers created via logging.getLogger() before init_logging()
    # might not properly inherit the root logger level that gets set here
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger_obj = logging.Logger.manager.loggerDict[logger_name]
        # Skip if it's a PlaceHolder (not yet instantiated)
        if not isinstance(logger_obj, logging.Logger):
            continue
        # Skip loggers we've explicitly set levels for (noisy ones and their children)
        # Only skip if we actually suppressed them (not when level_from_env is True)
        if not level_from_env and (logger_name in noisy_loggers or any(logger_name.startswith(f"{n}.") for n in noisy_loggers)):
            continue
        # Ensure propagation is enabled (allows inheritance from root)
        if not logger_obj.propagate:
            logger_obj.propagate = True
        # If logger has NOTSET level, it should inherit from root
        # We don't need to do anything else - Python's logging will handle inheritance
        # But we ensure propagate=True so messages bubble up to root handler

    _CONFIGURED = True


class _RunContextFilter(logging.Filter):
    """Inject a stable run_id into every record."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


class _ShortNameFormatter(logging.Formatter):
    """Formatter that shortens logger names to last 2 hierarchies."""

    def __init__(self, debug_fmt: str, normal_fmt: str, datefmt: str = None):
        """
        Initialize formatter with two formats.

        Args:
            debug_fmt: Full format string for DEBUG level (includes timestamp, run_id, full name)
            normal_fmt: Concise format string for non-DEBUG levels
            datefmt: Date format string
        """
        super().__init__(fmt=normal_fmt, datefmt=datefmt)
        self.debug_fmt = debug_fmt
        self.normal_fmt = normal_fmt
        self.datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with appropriate format based on level."""
        # Shorten logger name to last 2 hierarchies for non-DEBUG
        name_parts = record.name.split(".")
        if len(name_parts) > 2:
            short_name = ".".join(name_parts[-2:])
        else:
            short_name = record.name

        # Use full format for DEBUG, concise for others
        if record.levelno == logging.DEBUG:
            # Create a new formatter with debug format
            debug_formatter = logging.Formatter(fmt=self.debug_fmt, datefmt=self.datefmt)
            return debug_formatter.format(record)
        else:
            # Use concise format with short name
            original_name = record.name
            record.name = short_name
            # Create a new formatter with normal format
            normal_formatter = logging.Formatter(fmt=self.normal_fmt, datefmt=self.datefmt)
            result = normal_formatter.format(record)
            record.name = original_name
            return result


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    All configuration is done via init_logging() at the app entry point
    (or via autocfg on import). This function never touches handlers
    or levels; it just returns logging.getLogger(name).
    """
    return logging.getLogger(name)


def add_logging_cli_args(parser: Any) -> None:
    """Add --log-level/--debug/--log-file flags to an argparse parser.

    Typical usage:

        from hailo_logger import add_logging_cli_args, init_logging, level_from_args

        parser = argparse.ArgumentParser()
        add_logging_cli_args(parser)
        args = parser.parse_args()
        init_logging(level=level_from_args(args), log_file=args.log_file)
    """
    parser.add_argument(
        "--log-level",
        default=os.getenv("HAILO_LOG_LEVEL", "INFO"),
        choices=[k.lower() for k in _LEVELS.keys()],
        help="Logging level (default: %(default)s or $HAILO_LOG_LEVEL).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shortcut for DEBUG log level (overrides --log-level).",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("HAILO_LOG_FILE"),
        help="Optional log file path (also respects $HAILO_LOG_FILE).",
    )


def level_from_args(args: Any) -> str:
    """Resolve level string from argparse args."""
    return (
        "DEBUG"
        if getattr(args, "debug", False)
        else str(getattr(args, "log_level", "INFO")).upper()
    )


# Auto-configuration: set HAILO_LOG_AUTOCONFIG=1 to call init_logging() at import time.
# Level comes from HAILO_LOG_LEVEL env var (defaults to INFO).
# Noisy loggers are NOT suppressed when level is set via env var.
if os.getenv("HAILO_LOG_AUTOCONFIG", "0") == "1":
    try:
        init_logging()
    except Exception:
        # Avoid crashing on import due to logging config issues
        pass
