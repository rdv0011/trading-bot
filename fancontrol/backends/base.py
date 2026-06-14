from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PinConfig:
    """GPIO pin identity understood by every backend.

    Attributes:
        chip:  GPIO chip name, e.g. ``"gpiochip0"``, ``"gpiochip3"``.
        line:  Line offset within the chip, e.g. ``20``.
    """

    chip: str
    line: int


class GpioBackend(ABC):
    """Interface every GPIO backend implements.

    Subclasses **must** call ``register()`` at module level so the
    backend is discoverable by :func:`detect_backend`.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Human-readable backend identifier (lowercase, no spaces)."""

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Return True if this backend can run on the current system.

        Checks are typically ``shutil.which()`` for CLI tools, or
        ``try: import gpiod`` for Python libraries.
        """

    @abstractmethod
    def set_value(self, pin: PinConfig, value: bool) -> None:
        """Set a GPIO line high (True) or low (False)."""

    def cleanup(self) -> None:
        """Release any resources held by this backend.

        Base implementation is a no-op; override if the backend
        allocates persistent state (e.g. libgpiod line requests).
        """
