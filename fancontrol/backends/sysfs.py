"""Backend using the legacy sysfs GPIO interface (``/sys/class/gpio``).

Works on any Linux board with GPIO sysfs enabled in the kernel — no
extra packages required.  This is the **universal fallback** when
neither ``gpioset`` nor ``raspi-gpio`` nor ``pinctrl`` is available.

.. note::
    The sysfs interface is deprecated but widely supported on older
    kernels (3.x – 5.x).  Modern kernels may require
    ``gpio-sysfs`` kernel module or
    ``CONFIG_GPIO_SYSFS=y``.
"""

import os
from pathlib import Path

from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register


class SysfsBackend(GpioBackend):
    """Write to ``/sys/class/gpio/gpioN/value``"""

    @classmethod
    def name(cls) -> str:
        return "sysfs"

    @classmethod
    def available(cls) -> bool:
        return os.path.isdir("/sys/class/gpio")

    def set_value(self, pin: PinConfig, value: bool) -> None:
        gpio_dir = Path(f"/sys/class/gpio/gpio{pin.line}")
        if not gpio_dir.exists():
            Path("/sys/class/gpio/export").write_text(str(pin.line))
        (gpio_dir / "direction").write_text("out")
        (gpio_dir / "value").write_text("1" if value else "0")


register(SysfsBackend)
