"""Backend using the ``gpioset`` CLI tool from libgpiod v1.

Works on any Linux system with GPIO character device support and
the ``gpiod`` package installed.

.. code-block:: bash

    sudo apt install gpiod           # Debian / Ubuntu
    sudo dnf install libgpiod-tools  # Fedora
    sudo pacman -S gpiod             # Arch

The GPIO line state persists even after the calling process exits,
which is exactly what we want — no daemon required.
"""

import logging
import shutil

from fancontrol.backends._helpers import run_cli
from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register

logger = logging.getLogger(__name__)


class GpiosetBackend(GpioBackend):
    """``sudo gpioset -c <chip> <line>=<0|1>``"""

    @classmethod
    def name(cls) -> str:
        return "gpioset"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("gpioset") is not None

    def set_value(self, pin: PinConfig, value: bool) -> None:
        try:
            run_cli(
                [
                    "sudo",
                    "gpioset",
                    "-c",
                    pin.chip,
                    f"{pin.line}={1 if value else 0}",
                ],
                log_prefix="gpioset",
            )
        except RuntimeError:
            raise RuntimeError(
                f"gpioset failed for {pin.chip} line {pin.line}. "
                f"Check: (1) sudo passwordless rule for gpioset, "
                f"(2) '{pin.chip}' exists — run 'ls /dev/gpiochip*', "
                f"(3) line {pin.line} is available — run 'gpioinfo {pin.chip}'."
            )


register(GpiosetBackend)
