"""Shared utilities for CLI-based GPIO backends."""

import logging
import shlex
import subprocess

logger = logging.getLogger(__name__)


def run_cli(args: list[str], *, log_prefix: str = "gpio") -> None:
    """Run a CLI command and raise on failure.

    All CLI backends (gpioset, raspi-gpio, pinctrl) use this so
    errors are surfaced instead of swallowed silently.

    Args:
        args: Command and arguments (e.g. ``["sudo", "gpioset", ...]``).
        log_prefix: Short label for log messages.

    Raises:
        RuntimeError: The command exited with a non-zero code.
    """
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode == 0:
        return

    cmd_str = shlex.join(args)
    stderr = result.stderr.strip()
    detail = stderr if stderr else cmd_str
    msg = f"{log_prefix} command failed (exit {result.returncode}): {detail}"
    logger.error(msg)
    raise RuntimeError(msg)
