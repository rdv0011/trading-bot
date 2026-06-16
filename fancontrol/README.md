# Fan Control — CPU Cooling

A lightweight libgpiod v2 PWM fan controller that ramps fan speed
based on CPU temperature.  Designed for the **Radxa Zero 3W** but
adaptable to any Linux SBC with minor edits.

---

## What's Here

| File | Purpose |
|------|---------|
| `fan_pwm.py` | PWM fan controller script (source) |
| `fan-pwm.service` | systemd unit for automatic startup |

---

## Prerequisites

The controller uses `libgpiod` v2 Python bindings.  Install them:

```bash
sudo apt update && sudo apt install python3-libgpiod
```

---

## Hardware Wiring (Radxa Zero 3W)

A 5V two-wire fan is controlled via an N-channel MOSFET (e.g. IRFZ44N,
2N7000, or any logic-level MOSFET) connected to the 40-pin header.

### Pin Assignment

| Header Pin | Connects To          | Notes                     |
|------------|----------------------|---------------------------|
| **4**      | Fan + wire (red)    | +5V power                 |
| **6**      | MOSFET source → GND | Ground return             |
| **7**      | MOSFET gate         | GPIO control (3.3V logic) |

### Circuit Diagram

```
  Radxa 40-pin header
  ┌──────────────┐
  │  Pin 4 ──────┼────── Fan + (red)
  │  (+5V)       │
  │              │
  │  Pin 6 ──────┼────── MOSFET source ─── GND
  │  (GND)       │
  │              │
  │  Pin 7 ──┬───┼────── MOSFET gate (via 1kΩ resistor)
  │  (GPIO)  │   │
  └──────────┘   │
                 │
           ┌─────┴─────┐
           │   MOSFET  │
           │  (N-channel)
           │           │
           │  Drain ───┼── Fan - (black)
           │  Gate ────┘
           │  Source ──┘
           └───────────
```

### How It Works

- The MOSFET acts as a low-side switch — when pin 7 goes HIGH (3.3V),
  the MOSFET conducts and completes the fan's ground path, turning it on.
- The 1kΩ resistor between the GPIO pin and the MOSFET gate limits
  inrush current and protects the GPIO line.
- No flyback diode is needed for a **brushed** DC fan (it draws
  negligible inductive kickback).  For a **brushless** fan (most
  5V computer fans) the fan itself contains the driver circuitry.

---

## Quick Start (Radxa Zero 3W)

```bash
# 1. Deploy the script
sudo cp fancontrol/fan_pwm.py /usr/local/bin/fan_pwm.py
sudo chmod +x /usr/local/bin/fan_pwm.py

# 2. Deploy and enable the service
sudo cp fancontrol/fan-pwm.service /etc/systemd/system/fan-pwm.service
sudo systemctl daemon-reload
sudo systemctl enable fan-pwm.service
sudo systemctl start fan-pwm.service

# 3. Verify
sudo systemctl status fan-pwm.service
sudo journalctl -u fan-pwm.service -f
```

## Full Documentation

See **[`plans/update_fan_control.md`](../plans/update_fan_control.md)** for:

- Hardware verification steps
- Manual testing
- Temperature curve tuning
- Stress testing
- Service management commands
