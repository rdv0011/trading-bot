"""Backend using the ``pinctrl`` tool (Raspberry Pi 5 / BCM2712).

``pinctrl`` is the GPIO control tool shipped with Raspberry Pi 5
firmware.  It replaces ``raspi-gpio`` on Pi 5.

.. code-block:: bash

    sudo pinctrl set <line> dh   # drive high (fan on)
    sudo pinctrl set <line> dl   # drive low  (fan off)
"""

import logging
import shutil

from fancontrol.backends._helpers import run_cli
from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register

logger = logging.getLogger(__name__)


class PinctrlBackend(GpioBackend):
    """``sudo pinctrl set <line> dh|dl``"""

    @classmethod
    def name(cls) -> str:
        return "pinctrl"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("pinctrl") is not None

    def set_value(self, pin: PinConfig, value: bool) -> None:
        level = "dh" if value else "dl"
        try:
            run_cli(
                ["sudo", "pinctrl", "set", str(pin.line), level],
                log_prefix="pinctrl",
            )
        except RuntimeError:
            raise RuntimeError(
                f"pinctrl failed for line {pin.line}. "
                f"Check: (1) sudo passwordless rule, "
                f"(2) you are on a Raspberry Pi 5 (BCM2712)."
            )


register(PinctrlBackend)
