"""Backend using the ``raspi-gpio`` tool (Raspberry Pi OS classic).

Works on Raspberry Pi 2 / 3 / 4 / Zero (pre-Pi 5).  The tool is
pre-installed on Raspberry Pi OS.

.. code-block:: bash

    sudo raspi-gpio set <line> dh   # drive high (fan on)
    sudo raspi-gpio set <line> dl   # drive low  (fan off)
"""

import shutil
import subprocess

from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register


class RaspiGpioBackend(GpioBackend):
    """``sudo raspi-gpio set <line> dh|dl``"""

    @classmethod
    def name(cls) -> str:
        return "raspi-gpio"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("raspi-gpio") is not None

    def set_value(self, pin: PinConfig, value: bool) -> None:
        level = "dh" if value else "dl"
        subprocess.run(
            ["sudo", "raspi-gpio", "set", str(pin.line), level],
            check=False,
            capture_output=True,
        )


register(RaspiGpioBackend)
