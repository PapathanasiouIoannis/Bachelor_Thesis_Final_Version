import os
import logging
import sys
from pathlib import Path


_MANAGED_HANDLER_ATTRIBUTE = "_eoslab_managed_file_handler"
_ACTIVE_LOG_FILE: Path | None = None


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _default_log_file() -> Path:
    current_dir = Path(__file__).resolve().parent
    return current_dir.parents[1] / "pipeline_debug.log"


def _add_managed_file_handler(logger: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_formatter())
    setattr(file_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    logger.addHandler(file_handler)


def configure_run_log(path: str | os.PathLike) -> Path:
    """Route managed project loggers to one experiment-specific log file."""

    global _ACTIVE_LOG_FILE
    _ACTIVE_LOG_FILE = Path(path).resolve()
    for candidate in logging.Logger.manager.loggerDict.values():
        if not isinstance(candidate, logging.Logger):
            continue
        for handler in list(candidate.handlers):
            if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
                candidate.removeHandler(handler)
                handler.close()
        if candidate.name in {
            "SOLVE_SEQ",
            "VISUALIZATION",
            "TABLE_GEN",
            "PHYSICS_PIPELINE",
            "EOSLAB",
        }:
            _add_managed_file_handler(candidate, _ACTIVE_LOG_FILE)
    return _ACTIVE_LOG_FILE


def close_run_log() -> None:
    """Close managed file handlers in the current process."""

    global _ACTIVE_LOG_FILE
    for candidate in logging.Logger.manager.loggerDict.values():
        if not isinstance(candidate, logging.Logger):
            continue
        for handler in list(candidate.handlers):
            if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
                candidate.removeHandler(handler)
                handler.close()
    _ACTIVE_LOG_FILE = None


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger with the structured formatter:
    [TIMESTAMP] [MODULE_NAME] [LEVEL] - Message
    Outputs to both stdout and a persistent pipeline_debug.log file.
    """
    logger = logging.getLogger(module_name)

    # ``logger.handlers`` is intentional: a parent/root handler must not prevent
    # this project logger from receiving its own stable formatter.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(_formatter())
        logger.addHandler(stream_handler)
        _add_managed_file_handler(logger, _ACTIVE_LOG_FILE or _default_log_file())

    return logger


__all__ = ["close_run_log", "configure_run_log", "get_logger"]
