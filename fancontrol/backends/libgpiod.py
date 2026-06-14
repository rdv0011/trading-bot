"""Backend using the Python libgpiod v2 bindings (``gpiod`` pip package >= 2.0).

The v2 API is completely different from v1:

* ``gpiod.request_lines()`` replaces ``gpiod.Chip().get_line()``.
* ``LineRequest.set_value()`` replaces ``Line.set_value()``.
* ``gpiod.line.Value.ACTIVE / .INACTIVE`` enum replaces plain ``int``.
"""

from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register


class LibgpiodBackend(GpioBackend):
    """Python :mod:`gpiod` v2 bindings — fastest option."""

    _requests: dict[str, "gpiod.LineRequest"] = {}

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
        from gpiod.line import Direction, Value as LineValue

        key = f"{pin.chip}:{pin.line}"
        if key not in self._requests:
            chip_path = (
                pin.chip
                if pin.chip.startswith("/dev/")
                else f"/dev/{pin.chip}"
            )
            request = gpiod.request_lines(
                chip_path,
                consumer="fanctl",
                config={
                    pin.line: gpiod.LineSettings(
                        direction=Direction.OUTPUT,
                        output_value=LineValue.INACTIVE,
                    )
                },
            )
            self._requests[key] = request
        self._requests[key].set_value(
            pin.line, LineValue.ACTIVE if value else LineValue.INACTIVE
        )

    def cleanup(self) -> None:
        for request in self._requests.values():
            try:
                request.release()
            except Exception:
                pass
        self._requests.clear()


register(LibgpiodBackend)
