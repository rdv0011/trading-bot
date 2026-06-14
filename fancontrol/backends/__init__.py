"""GPIO backend registry and auto-detection.

Usage::

    from fancontrol.backends import detect_backend

    backend = detect_backend()           # auto-detect best available
    backend = detect_backend("gpioset")  # force specific backend
"""

import logging
from typing import Type

from fancontrol.backends.base import GpioBackend

_BUILTIN_BACKENDS: list[Type[GpioBackend]] = []


def register(backend_cls: Type[GpioBackend]) -> None:
    """Register a :class:`GpioBackend` subclass for auto-detection."""
    _BUILTIN_BACKENDS.append(backend_cls)


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return [cls.name() for cls in _BUILTIN_BACKENDS]


# ── Priority order (most capable first) ──────────────────────────
_PRIORITY = ["libgpiod", "gpioset", "pinctrl", "raspi-gpio", "sysfs"]


def detect_backend(preferred: str | None = None) -> GpioBackend:
    """Detect the best available GPIO backend.

    The detection order is:

        1. **libgpiod** — Python bindings (fastest, no subprocess).
        2. **gpioset**  — libgpiod v1 CLI tool (widely available).
        3. **pinctrl**  — Raspberry Pi 5 built-in tool.
        4. **raspi-gpio** — Raspberry Pi OS classic tool.
        5. **sysfs**    — Legacy ``/sys/class/gpio`` (universal fallback).

    Args:
        preferred: If given, force a specific backend by name.
            Raises :class:`RuntimeError` if the named backend is
            not available on this system.

    Returns:
        An instantiated :class:`GpioBackend`.

    Raises:
        RuntimeError: No backend is available on this system.
    """
    if preferred:
        for cls in _BUILTIN_BACKENDS:
            if cls.name() == preferred:
                if cls.available():
                    return cls()
                raise RuntimeError(
                    f"Preferred backend '{preferred}' is not available "
                    f"on this system. Available: {list_backends()}"
                )
        raise RuntimeError(
            f"Unknown backend '{preferred}'. Known backends: {list_backends()}"
        )

    for name in _PRIORITY:
        for cls in _BUILTIN_BACKENDS:
            if cls.name() == name and cls.available():
                logging.info("FanControl: auto-detected backend '%s'", name)
                return cls()

    raise RuntimeError(
        "No GPIO backend available on this system.\n"
        "  - Install gpiod (apt install gpiod) for gpioset support.\n"
        "  - Or install python3-libgpiod (apt install python3-libgpiod).\n"
        "  - Or check that /sys/class/gpio exists (legacy sysfs).\n"
        "See fancontrol/README.md for board-specific setup."
    )


# ── Discover built-in backends at import time ────────────────────


def _discover_backends() -> None:
    import fancontrol.backends.gpioset  # noqa: F401
    import fancontrol.backends.raspi_gpio  # noqa: F401
    import fancontrol.backends.pinctrl  # noqa: F401
    import fancontrol.backends.sysfs  # noqa: F401
    try:
        import fancontrol.backends.libgpiod  # noqa: F401
    except ImportError:
        pass  # optional — gpiod package not installed


_discover_backends()
