"""Central application logging configuration for GGUI."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def get_logger(component: str) -> logging.Logger:
    """Return a component logger under the ``ggui`` namespace."""
    configure_logging()
    return logging.getLogger(f"ggui.{component}")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent standard logging setup for the application."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("ggui")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    _CONFIGURED = True


def log_operation(
    component: str,
    stage: str,
    *,
    case_dir: Optional[str] = None,
    safe_command: Optional[str] = None,
    exit_code: Optional[int] = None,
    exc: Optional[BaseException] = None,
    level: int = logging.INFO,
) -> None:
    logger = get_logger(component)
    parts = [stage]
    if case_dir:
        parts.append(f"case={case_dir}")
    if safe_command:
        parts.append(f"cmd={safe_command}")
    if exit_code is not None:
        parts.append(f"exit={exit_code}")
    message = " | ".join(parts)
    if exc is not None:
        logger.log(level, message, exc_info=exc)
    else:
        logger.log(level, message)
