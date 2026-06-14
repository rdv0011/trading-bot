"""Backend using the ``pinctrl`` tool (Raspberry Pi 5 / BCM2712).

``pinctrl`` is the GPIO control tool shipped with Raspberry Pi 5
firmware.  It replaces ``raspi-gpio`` on Pi 5.

.. code-block:: bash

    sudo pinctrl set <line> dh   # drive high (fan on)
    sudo pinctrl set <line> dl   # drive low  (fan off)
"""

import shutil
import subprocess

from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register


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
        subprocess.run(
            ["sudo", "pinctrl", "set", str(pin.line), level],
            check=False,
            capture_output=True,
        )


register(PinctrlBackend)
