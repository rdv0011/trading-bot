"""High-level fan controller with context manager and temperature monitoring.

Typical usage::

    from fancontrol.fanctl import fan_control

    # Automatically turns fan ON before the block and OFF after,
    # even if the block raises an exception or the process receives
    # SIGTERM / SIGINT (Ctrl+C).

    with fan_control(enable=True, temp_threshold=60.0):
        run_training(...)

For manual control::

    from fancontrol.fanctl import FanController

    ctrl = FanController()
    ctrl.on()
    ...
    ctrl.off()
"""

import atexit
import logging
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from fancontrol.backends import detect_backend
from fancontrol.backends.base import GpioBackend
from fancontrol.config import FanConfig, load_config

logger = logging.getLogger(__name__)

# Ensure log messages are visible without requiring users to configure logging.
# This only installs a handler if none exists, so user-configured logging
# (e.g. via ``logging.basicConfig`` in main.py) takes priority.
if not logger.handlers and not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


# ── CPU temperature helpers ─────────────────────────────────────


def _read_cpu_temp() -> Optional[float]:
    """Read CPU temperature from Linux thermal zones.

    Returns:
        Degrees Celsius, or ``None`` if no thermal zone is available
        (e.g. macOS, or kernel without thermal support).
    """
    zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    for zone in zones:
        try:
            raw = int(zone.read_text().strip())
            return raw / 1000.0
        except (OSError, ValueError):
            continue
    return None


# ── FanController ───────────────────────────────────────────────


class FanController:
    """GPIO fan controller with temperature-aware auto-control.

    The GPIO line state set by :meth:`on` **persists** even after the
    process exits (``gpioset`` is stateless — the kernel holds the
    line).  The :meth:`off` method and signal handlers exist to restore
    the system to a known state on clean shutdown.

    Args:
        config: Configuration.  Loaded from environment / file if
            ``None``.
    """

    def __init__(self, config: Optional[FanConfig] = None):
        self._config = config or load_config()
        self._backend: GpioBackend = detect_backend(self._config.backend)
        self._is_on = False
        logger.info(
            "FanControl: config — chip=%s line=%d active_low=%s backend=%s "
            "temp_threshold=%.1f°C hysteresis=%.1f°C",
            self._config.pin.chip,
            self._config.pin.line,
            self._config.active_low,
            self._config.backend or self._backend.name(),
            self._config.temp_threshold,
            self._config.temp_hysteresis,
        )

    # -- read-only properties -----------------------------------------------

    @property
    def is_on(self) -> bool:
        """Whether the controller believes the fan is currently on."""
        return self._is_on

    @property
    def config(self) -> FanConfig:
        """Current configuration (read-write before :meth:`on`)."""
        return self._config

    # -- public API ---------------------------------------------------------

    def on(self) -> None:
        """Turn the fan on immediately.  Idempotent — safe to call twice.

        Raises:
            RuntimeError: The GPIO backend command failed (e.g. sudo
                requires a password, wrong chip/line, or permissions).
                Check the error message and your configuration.
        """
        if self._is_on:
            return
        value = not self._config.active_low
        try:
            self._backend.set_value(self._config.pin, value)
        except RuntimeError:
            self._is_on = False
            logger.error(
                "✗ Fan FAILED to turn on  (chip=%s line=%d). "
                "Verify: (1) sudo passwordless rule, "
                "(2) chip exists — 'ls /dev/gpiochip*', "
                "(3) line is correct — 'gpioinfo %s'.",
                self._config.pin.chip,
                self._config.pin.line,
                self._config.pin.chip,
            )
            raise
        self._is_on = True
        logger.info(
            "Fan ON  (chip=%s line=%d active_low=%s)",
            self._config.pin.chip,
            self._config.pin.line,
            self._config.active_low,
        )

    def off(self) -> None:
        """Turn the fan off immediately.  Idempotent."""
        if not self._is_on:
            return
        value = self._config.active_low
        try:
            self._backend.set_value(self._config.pin, value)
        except RuntimeError:
            logger.error(
                "✗ Fan FAILED to turn off (chip=%s line=%d). "
                "The GPIO line may be stuck.  Run manually: "
                "'sudo gpioset -c %s %d=0'.",
                self._config.pin.chip,
                self._config.pin.line,
                self._config.pin.chip,
                self._config.pin.line,
            )
            self._is_on = False  # mark off regardless — state is uncertain
            return
        self._is_on = False
        logger.info(
            "Fan OFF (chip=%s line=%d)",
            self._config.pin.chip,
            self._config.pin.line,
        )

    def auto(self) -> None:
        """Check CPU temperature and toggle fan based on threshold.

        - If temperature exceeds ``temp_threshold`` → fan ON.
        - If temperature drops below ``temp_threshold - hysteresis`` → fan OFF.
        - If ``temp_threshold`` is ``0`` (disabled) → no-op.

        This method is **non-blocking** — it reads the temperature once
        and returns immediately.
        """
        threshold = self._config.temp_threshold
        if threshold <= 0:
            logger.info(
                "FanControl: temp_threshold=%.1f — auto-control disabled "
                "(set FAN_TEMP_THRESHOLD or temp_threshold in fanctl.toml)",
                threshold,
            )
            return

        temp = _read_cpu_temp()
        if temp is None:
            logger.info(
                "FanControl: no CPU temp sensor found — cannot auto-control. "
                "Run 'ls /sys/class/thermal/' to check available zones, "
                "or call .on() directly to force the fan on."
            )
            return

        hysteresis = self._config.temp_hysteresis
        if self._is_on:
            if temp < (threshold - hysteresis):
                logger.info(
                    "FanControl: CPU %.1f°C < %.1f°C — turning fan OFF",
                    temp,
                    threshold - hysteresis,
                )
                self.off()
        else:
            if temp > threshold:
                logger.info(
                    "FanControl: CPU %.1f°C > %.1f°C — turning fan ON",
                    temp,
                    threshold,
                )
                self.on()

    def cleanup(self) -> None:
        """Release backend resources (e.g. libgpiod line requests).

        Also ensures the fan is turned off.
        """
        self.off()
        self._backend.cleanup()


# ── Module-level singleton + crash safety ───────────────────────


_controller: Optional[FanController] = None


def _get_controller(config: Optional[FanConfig] = None) -> FanController:
    global _controller
    if _controller is None:
        _controller = FanController(config)
    return _controller


def _shutdown(*_args) -> None:
    """Signal handler / atexit callback — turn fan off and release GPIO."""
    global _controller
    if _controller is not None:
        _controller.off()
        _controller.cleanup()


def _register_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            pass  # not running in the main thread


_register_signal_handlers()
atexit.register(_shutdown)


# ── Context manager ─────────────────────────────────────────────


@contextmanager
def fan_control(
    *,
    enable: bool = True,
    temp_threshold: Optional[float] = None,
    config: Optional[FanConfig] = None,
) -> Generator[FanController, None, None]:
    """Context manager: fan ON during the block, OFF after (even on crash).

    Args:
        enable:
            Set to ``False`` to skip all fan control — useful when the
            feature is toggled off at the command line.
        temp_threshold:
            Override the temperature threshold for this session, in °C.
            ``0`` disables thermal auto-control.  ``None`` keeps the
            value from ``config`` / environment.
        config:
            Full :class:`FanConfig` override.  If not provided, the
            module-level singleton configuration is used (loaded from
            environment variables or ``fanctl.toml``).

    Yields:
        The :class:`FanController` instance so callers can call
        ``.auto()`` manually inside the block if desired.
    """
    if not enable:
        yield _get_controller(config) if config else FanController()
        return

    ctrl = FanController(config) if config else _get_controller()

    if temp_threshold is not None:
        ctrl._config.temp_threshold = temp_threshold

    ctrl.on()
    try:
        yield ctrl
    finally:
        ctrl.off()
