"""Backend using the Python libgpiod bindings (``gpiod`` package).

This backend talks to the GPIO character device directly — no
subprocess, no ``sudo`` requirement (if the user has appropriate
permissions or ``udev`` rules).

.. code-block:: bash

    # Debian / Ubuntu
    sudo apt install python3-libgpiod

    # Via pip (requires libgpiod-dev)
    pip install gpiod

The line requests are cached so that repeated calls to the same
pin do not re-open the chip each time.
"""

from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register


class LibgpiodBackend(GpioBackend):
    """Python :mod:`gpiod` bindings — fastest option."""

    _lines: dict[str, "gpiod.Line"] = {}

    @classmethod
    def name(cls) -> str:
        return "libgpiod"

    @classmethod
    def available(cls) -> bool:
        try:
            import gpiod  # noqa: F401
            return True
        except ImportError:
            return False

    def set_value(self, pin: PinConfig, value: bool) -> None:
        import gpiod

        key = f"{pin.chip}:{pin.line}"
        if key not in self._lines:
            # gpiod v2 expects full /dev/gpiochipN path (not just gpiochipN)
            chip_path = pin.chip if pin.chip.startswith("/dev/") else f"/dev/{pin.chip}"
            chip = gpiod.Chip(chip_path)
            line = chip.get_line(pin.line)
            line.request(consumer="fanctl", type=gpiod.LINE_REQ_DIR_OUT)
            self._lines[key] = line
        self._lines[key].set_value(int(value))

    def cleanup(self) -> None:
        for line in self._lines.values():
            try:
                line.release()
            except Exception:
                pass
        self._lines.clear()


register(LibgpiodBackend)
