"""GPIO-based fan control for CPU cooling during compute workloads.

Quick start::

    from fancontrol.fanctl import fan_control

    with fan_control(enable=True):
        run_training()

See ``fancontrol/README.md`` for board configuration and adaptation.
"""

from fancontrol.fanctl import FanController, fan_control
from fancontrol.config import FanConfig, load_config
from fancontrol.backends.base import PinConfig, GpioBackend
from fancontrol.backends import detect_backend, list_backends

__all__ = [
    "FanController",
    "FanConfig",
    "PinConfig",
    "GpioBackend",
    "fan_control",
    "load_config",
    "detect_backend",
    "list_backends",
]
