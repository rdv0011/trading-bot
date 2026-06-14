"""Configuration loading for the fan control system.

Supports two sources (later takes precedence):

    1. **TOML file** — ``fanctl.toml`` (or custom path via ``FAN_CONFIG`` env var).
    2. **Environment variables** — overrides individual fields.

Example ``fanctl.toml``::

    [fan]
    chip = "gpiochip0"
    line = 18
    active_low = false
    backend = "gpioset"
    temp_threshold = 60.0
    temp_hysteresis = 5.0

Available environment variables:

    ====================== ====================== ===========
    Variable               Default                Description
    ====================== ====================== ===========
    ``FAN_GPIO_CHIP``      ``gpiochip0``          GPIO chip
    ``FAN_GPIO_LINE``      ``20``                 GPIO line
    ``FAN_ACTIVE_LOW``     ``false``              Invert logic
    ``FAN_BACKEND``        *(auto-detect)*        Force backend
    ``FAN_TEMP_THRESHOLD`` ``0``                  °C (0=off)
    ``FAN_TEMP_HYSTERESIS`` ``5.0``               °C deadband
    ====================== ====================== ===========
"""

import os
from dataclasses import dataclass, field

from fancontrol.backends.base import PinConfig


@dataclass
class FanConfig:
    """Complete fan control configuration.

    Attributes:
        pin:
            GPIO pin identity (chip + line).
        active_low:
            If ``True``, the fan runs when the GPIO is LOW (electrically
            inverted circuit).  Default: ``False``.
        backend:
            Preferred backend name, or ``None`` for auto-detection.
        temp_threshold:
            CPU temperature in °C above which the fan turns on
            automatically.  ``0`` (the default) disables thermal auto-control.
        temp_hysteresis:
            Temperature deadband in °C.  Once the fan is on, the
            temperature must drop *below* ``temp_threshold - hysteresis``
            before the fan turns off.  Prevents rapid on/off cycling.
    """

    pin: PinConfig = field(default_factory=lambda: PinConfig("gpiochip0", 20))
    active_low: bool = False
    backend: str | None = None
    temp_threshold: float = 0.0
    temp_hysteresis: float = 5.0


def load_config(path: str | None = None) -> FanConfig:
    """Load configuration from TOML file + environment variables.

    Resolution order (later wins):

        1. Hard-coded defaults.
        2. Values from the TOML ``[fan]`` section (if file exists).
        3. Individual environment variables.

    Args:
        path: Path to a TOML config file.  Falls back to the
            ``FAN_CONFIG`` env var if ``None``, then to ``fanctl.toml``
            in the current directory.
    """
    chip = "gpiochip0"
    line = 20
    active_low = False
    backend: str | None = None
    temp_threshold = 0.0
    temp_hysteresis = 5.0

    # ── 1. TOML config file ─────────────────────────────────
    if path is None:
        path = os.getenv("FAN_CONFIG", "fanctl.toml")

    if path and os.path.exists(path):
        try:
            import tomllib as _toml  # Python ≥3.11
        except ImportError:
            try:
                import tomli as _toml  # pip install tomli
            except ImportError:
                _toml = None  # no TOML support — skip file

        if _toml is not None:
            with open(path, "rb") as f:
                data = _toml.load(f).get("fan", {})
            chip = data.get("chip", chip)
            line = int(data.get("line", line))
            active_low = bool(data.get("active_low", active_low))
            backend = data.get("backend", backend)
            temp_threshold = float(data.get("temp_threshold", temp_threshold))
            temp_hysteresis = float(data.get("temp_hysteresis", temp_hysteresis))

    # ── 2. Environment variable overrides ───────────────────
    chip = os.getenv("FAN_GPIO_CHIP", chip)
    line = int(os.getenv("FAN_GPIO_LINE", str(line)))
    active_low = os.getenv("FAN_ACTIVE_LOW", str(active_low)).lower() in ("true", "1", "yes")
    backend = os.getenv("FAN_BACKEND", backend or "") or None
    temp_threshold = float(os.getenv("FAN_TEMP_THRESHOLD", str(temp_threshold)))
    temp_hysteresis = float(os.getenv("FAN_TEMP_HYSTERESIS", str(temp_hysteresis)))

    return FanConfig(
        pin=PinConfig(chip=chip, line=line),
        active_low=active_low,
        backend=backend,
        temp_threshold=temp_threshold,
        temp_hysteresis=temp_hysteresis,
    )
