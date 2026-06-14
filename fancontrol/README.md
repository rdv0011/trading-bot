# Fan Control — GPIO CPU Cooling

Fans the CPU during simulation-driven training to prevent thermal
throttling.  Designed to be **board-agnostic**: it auto-detects the
GPIO interface available on your system and works on any Linux SBC.

---

## Quick Start

```bash
# 1. Find your fan header's GPIO chip and line
ls /dev/gpiochip*

# Radxa Zero 3W — gpiochip3 line 20 (physical pin 7)
sudo gpioset -c gpiochip3 20=1   # test — does the fan spin?
sudo gpioset -c gpiochip3 20=0   # test — does it stop?

# 2. Configure (via env vars or TOML)
export FAN_GPIO_CHIP=gpiochip3
export FAN_GPIO_LINE=20

# 3. Run training with fan control
python main.py --train-strategic --optimize-params --fan-control
```

For a quick test with temperature-based auto-control:

```bash
export FAN_TEMP_THRESHOLD=30    # °C — triggers on any running CPU
python main.py --train-strategic --fan-control
```

---

## Usage

### Via CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--fan-control` | off | Enable GPIO fan control |
| `--fan-temp-threshold` | config | °C threshold; set low (30) for testing |

These flags are available on:

- `python main.py --train-strategic --fan-control`
- `python strategic/strategictraining.py --fan-control`
- `python dualmlsimulation.py --fan-control`

### Via environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FAN_GPIO_CHIP` | `gpiochip0` | GPIO chip device name |
| `FAN_GPIO_LINE` | `20` | GPIO line offset within the chip |
| `FAN_ACTIVE_LOW` | `false` | Set `true` if fan runs when GPIO is LOW |
| `FAN_BACKEND` | *(auto)* | Force a specific backend |
| `FAN_TEMP_THRESHOLD` | `0` (off) | CPU temperature °C to auto-start fan |
| `FAN_TEMP_HYSTERESIS` | `5.0` | °C deadband before fan turns off |
| `FAN_CONFIG` | `fanctl.toml` | Path to TOML config file |

### Via TOML config file

Copy to the project root (alongside ``main.py``).  The file is found
automatically — no environment variables or flags needed.

```bash
cp fancontrol/fanctl.toml.example fanctl.toml
nano fanctl.toml              # adjust chip, line, threshold, etc.
python main.py --train-strategic --fan-control
```

Search order (``load_config`` resolves the first match):

1. ``FAN_CONFIG`` environment variable pointing to a specific path.
2. ``fanctl.toml`` in the project root (parent of ``fancontrol/``).
3. ``fanctl.toml`` in the current working directory.

You can verify which file was loaded by watching for the log line:

```
FanControl: loading config from /home/armbian/trading-bot/fanctl.toml
```

### Programmatic (in Python)

```python
from fancontrol.fanctl import fan_control, FanController

# Context manager — fan ON during block, OFF after (even on crash)
with fan_control(enable=True):
    run_training()

# Manual control
ctrl = FanController()
ctrl.on()
do_heavy_compute()
ctrl.off()

# Temperature-aware (auto on/off based on CPU temp)
import time
ctrl = FanController()
ctrl.config.temp_threshold = 60.0  # °C
while training:
    ctrl.auto()    # checks temp, toggles fan if needed
    time.sleep(30)
```

---

## How It Works

The system is split into three layers:

```
┌─────────────────────────────────────────────────┐
│  fanctl.py          Context manager + signals   │
│  config.py          TOML + env var loading      │
├─────────────────────────────────────────────────┤
│  backends/                                       │
│    base.py          Abstract GpioBackend         │
│    __init__.py      Registry + auto-detection    │
│    gpioset.py       gpioset CLI (libgpiod v1)    │
│    raspi_gpio.py    raspi-gpio (Pi 2-4)          │
│    pinctrl.py       pinctrl (Pi 5)               │
│    sysfs.py         /sys/class/gpio (fallback)   │
│    libgpiod.py      Python gpiod bindings        │
└─────────────────────────────────────────────────┘
```

1. **Auto-detection** tries backends in order: `libgpiod` (fastest) →
   `gpioset` → `pinctrl` → `raspi-gpio` → `sysfs` (universal fallback).
2. The selected backend calls the appropriate tool or library to
   set the GPIO line.
3. **Signal handlers** (`SIGTERM`, `SIGINT`) and an `atexit` hook
   ensure the fan is turned off on clean exit, Ctrl+C, or crash.
4. The GPIO line **persists** in the kernel — even if the process
   is killed with `SIGKILL`, the fan stays in its last state.
   This is why the signal handlers exist: to restore a known state
   during graceful shutdown.

---

## Backend Reference

| Backend | Tool / Library | When Available |
|---------|---------------|----------------|
| `libgpiod` | Python `gpiod` module | `pip install gpiod` or `apt install python3-libgpiod` |
| `gpioset` | `gpioset` CLI | `apt install gpiod` (Debian) / `pacman -S gpiod` (Arch) |
| `pinctrl` | `pinctrl` CLI | Raspberry Pi 5 (built-in) |
| `raspi-gpio` | `raspi-gpio` CLI | Raspberry Pi OS (pre-installed on Pi 2-4) |
| `sysfs` | `/sys/class/gpio` | Any Linux kernel with `CONFIG_GPIO_SYSFS` |

---

## Sudo Configuration

All CLI-based backends require `sudo` to access GPIO.  Set up a
passwordless sudo rule for the specific tool:

```bash
# /etc/sudoers.d/fanctl
yourusername ALL=(ALL) NOPASSWD: /usr/bin/gpioset
```

Or, for the Python libgpiod backend, grant the user direct GPIO
access via `udev` (no sudo needed):

```bash
# 1. Create the gpio group if it doesn't exist (Armbian, some Armbian
#    derivatives, and minimal builds often omit it).
sudo groupadd gpio 2>/dev/null; echo "ok"

# 2. Add your user to the group.
sudo usermod -aG gpio $USER

# 3. Create a udev rule that grants the gpio group read-write access
#    to all GPIO chip devices.
echo 'SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/99-gpio.rules

# 4. Reload udev and log out / back in (or reboot).
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "Now log out and back in, or reboot."
```

After re-login, both the ``libgpiod`` Python bindings and the
``gpioset`` CLI work without ``sudo``.  The ``libgpiod`` backend is
preferred by auto-detection because it is faster (no subprocess) and
avoids sudo entirely.

---

## Adapting to a New Board

### 4-Step Process

**Step 1 — Discover GPIO resources**

```bash
# List GPIO chips
ls /dev/gpiochip*

# Inspect chip capabilities
sudo gpioinfo gpiochip0    # shows lines and current values

# Check kernel GPIO debugfs (alternative)
sudo cat /sys/kernel/debug/gpio
```

**Step 2 — Determine chip and line for your fan header**

GPIO numbering is board-specific.  Common mappings:

| Architecture | Example | Finding the Line |
|---|---|---|
| Amlogic A311D2 (Radxa Zero 3W) | `gpiochip3`, line 20 | Pin 7 on 40-pin header; `gpioinfo gpiochip3` |
| BCM (Raspberry Pi) | `gpiochip0`, line 18 | Use BCM GPIO number directly |
| Rockchip (Orange Pi, Rock Pi) | `gpiochip0`, line = bank×32 + pin | Check schematic or DTS |
| Allwinner (Orange Pi Zero) | `gpiochip0`, line varies | `cat /sys/kernel/debug/gpio` |
| Nvidia Tegra (Jetson) | `gpiochip0`, line = 396+ | Jetson pinmux spreadsheet |
| Amlogic (Odroid) | `gpiochip0`, line = 400-500 | `gpioinfo gpiochip0` |
| x86 Super I/O | `gpiochip1` or higher | `ls /dev/gpiochip*` + board manual |

**Step 3 — Test your pin manually**

```bash
# Radxa Zero 3W — gpiochip3 line 20 (physical pin 7):
sudo gpioset -c gpiochip3 20=1 && echo "ON" && sleep 2 && sudo gpioset -c gpiochip3 20=0 && echo "OFF"

# Raspberry Pi — BCM GPIO 18 (physical pin 12):
sudo gpioset -c gpiochip0 18=1 && echo "ON" && sleep 2 && sudo gpioset -c gpiochip0 18=0 && echo "OFF"
```

If `gpioset` is not available, try `pinctrl` or `raspi-gpio`:

```bash
sudo pinctrl set 18 dh && sleep 2 && sudo pinctrl set 18 dl
```

**Step 4 — Configure**

```bash
# Radxa Zero 3W:
export FAN_GPIO_CHIP=gpiochip3
export FAN_GPIO_LINE=20

# Raspberry Pi:
export FAN_GPIO_CHIP=gpiochip0
export FAN_GPIO_LINE=18
```

Or create `fanctl.toml`:

```toml
[fan]
# Radxa Zero 3W — gpiochip3 line 20 (physical pin 7)
chip = "gpiochip3"
line = 20
active_low = false
backend = "gpioset"
```

### If Your Board Needs a New Backend

Write a class that implements `GpioBackend` and register it:

```python
# fancontrol/backends/my_board.py
import shutil, subprocess
from fancontrol.backends.base import GpioBackend, PinConfig
from fancontrol.backends import register

class MyBoardBackend(GpioBackend):
    @classmethod
    def name(cls) -> str:
        return "my-board-tool"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("my-gpio-tool") is not None

    def set_value(self, pin: PinConfig, value: bool) -> None:
        subprocess.run(
            ["my-gpio-tool", "set", str(pin.line), "1" if value else "0"],
            check=False,
        )

register(MyBoardBackend)
```

The new backend is automatically discovered on the next import of
`fancontrol.backends`.

---

## Testing

### Test without hardware (dry mode)

If no GPIO hardware is available, the `sysfs` backend will not be
available either, and fan control will raise a clear error.  To test
the integration logic without real hardware, mock the backend:

```bash
# On macOS or non-GPIO Linux, the code gracefully skips
# when --fan-control is not passed.
python main.py --train-strategic  # runs without fan control

# With --fan-control it will raise:
#   RuntimeError: No GPIO backend available on this system
```

### Test with temperature threshold

Set a low threshold to trigger the fan immediately:

```bash
export FAN_TEMP_THRESHOLD=30
python main.py --train-strategic --fan-control
```

On most systems the CPU idles above 30°C, so the fan will turn on
within seconds.

### Test temperature auto-control manually

```bash
FAN_TEMP_THRESHOLD=40 python -c "
from fancontrol.fanctl import FanController
import time
ctrl = FanController()
for _ in range(5):
    ctrl.auto()
    print(f'Fan on: {ctrl.is_on}')
    time.sleep(2)
ctrl.off()
"
```

---

## Safety

- **Crash safety**: `atexit` and signal handlers (`SIGTERM`, `SIGINT`)
  turn the fan off on clean shutdown.
- **Hard kill**: If the process receives `SIGKILL`, the GPIO line
  retains its last state (kernel-level persistence).  The fan stays
  on.  Run `python -c "from fancontrol.fanctl import FanController; FanController().off()"`
  to recover, or use the shell wrapper.
- **Thermal runaway prevention**: The temperature threshold feature
  (`FAN_TEMP_THRESHOLD`) provides an independent safety layer —
  even if the context manager fails, a separate process monitoring
  temperature can turn the fan on.
