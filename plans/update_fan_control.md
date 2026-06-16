# Radxa Zero 3W GPIO PWM Fan Setup (Armbian)

This guide configures a 5V 2-wire fan connected through a MOSFET and controlled by GPIO line 20 on `gpiochip3` (PIN_7).

The solution uses:

* Python 3
* libgpiod v2
* A systemd service
* Software PWM based on CPU temperature

---

# 1. Verify GPIO Fan Hardware

Confirm that the fan can be manually controlled:

Turn fan ON:

```bash
sudo gpioset -c gpiochip3 20=1
```

Turn fan OFF:

```bash
sudo gpioset -c gpiochip3 20=0
```

If the fan responds correctly, continue.

---

# 2. Install Required Packages

Update package lists:

```bash
sudo apt update
```

Install Python GPIO library:

```bash
sudo apt install python3-libgpiod
```

Install stress testing utility (optional):

```bash
sudo apt install stress-ng
```

---

# 3. Create Fan Controller Script

Create the script:

```bash
sudo nano /usr/local/bin/fan_pwm.py
```

Paste the following:

```python
#!/usr/bin/env python3

import time
import gpiod

GPIO_CHIP = "/dev/gpiochip3"
GPIO_LINE = 20

# PWM settings
PWM_FREQ = 25.0
PERIOD = 1.0 / PWM_FREQ

# Fan characteristics
MIN_DUTY = 0.25
START_BOOST_TIME = 0.5

# Temperature curve (°C, duty)
CURVE = [
    (50.0, 0.00),
    (55.0, 0.25),
    (60.0, 0.35),
    (65.0, 0.50),
    (70.0, 0.75),
    (75.0, 1.00),
]

def read_temp():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000.0

def temp_to_duty(temp):
    if temp <= CURVE[0][0]:
        return CURVE[0][1]

    for i in range(len(CURVE) - 1):
        t1, d1 = CURVE[i]
        t2, d2 = CURVE[i + 1]

        if temp <= t2:
            duty = d1 + (d2 - d1) * (temp - t1) / (t2 - t1)

            if duty > 0:
                duty = max(MIN_DUTY, duty)

            return duty

    return 1.0

def print_status(temp, duty):
    print(
        f"TEMP={temp:5.1f}°C  "
        f"DUTY={duty*100:5.1f}%"
    )

with gpiod.request_lines(
    GPIO_CHIP,
    consumer="fan-pwm",
    config={
        GPIO_LINE: gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT
        )
    },
) as req:

    print("Fan PWM controller started")

    last_duty = -1.0
    fan_running = False

    try:
        while True:
            temp = read_temp()
            duty = temp_to_duty(temp)

            if last_duty < 0 or abs(duty - last_duty) >= 0.05:
                print_status(temp, duty)
                last_duty = duty

            if duty <= 0:
                if fan_running:
                    print("Fan OFF")
                    fan_running = False

                req.set_value(GPIO_LINE, gpiod.line.Value.INACTIVE)
                time.sleep(2)
                continue

            if not fan_running:
                print("Fan START BOOST")
                req.set_value(GPIO_LINE, gpiod.line.Value.ACTIVE)
                time.sleep(START_BOOST_TIME)
                fan_running = True

            if duty >= 0.99:
                req.set_value(GPIO_LINE, gpiod.line.Value.ACTIVE)
                time.sleep(2)
                continue

            on_time = PERIOD * duty
            off_time = PERIOD - on_time

            req.set_value(GPIO_LINE, gpiod.line.Value.ACTIVE)
            time.sleep(on_time)

            req.set_value(GPIO_LINE, gpiod.line.Value.INACTIVE)
            time.sleep(off_time)

    except KeyboardInterrupt:
        print("\nStopping fan controller...")

    finally:
        req.set_value(GPIO_LINE, gpiod.line.Value.INACTIVE)
        print("Fan OFF")
```

Save and exit.

---

# 4. Make Script Executable

```bash
sudo chmod +x /usr/local/bin/fan_pwm.py
```

---

# 5. Test Manually

Run:

```bash
sudo python3 /usr/local/bin/fan_pwm.py
```

You should see output similar to:

```text
Fan PWM controller started
TEMP=52.0°C DUTY=0.0%
TEMP=58.0°C DUTY=29.0%
Fan START BOOST
```

Stop with:

```text
Ctrl+C
```

---

# 6. Stress Test

In a second terminal:

```bash
stress-ng --cpu 0 --timeout 300s
```

Monitor temperature:

```bash
watch -n 1 'echo "$(($(cat /sys/class/thermal/thermal_zone0/temp)/1000)) °C"'
```

Observe the fan speed increasing as temperature rises.

---

# 7. Configure Automatic Startup

Create a systemd service:

```bash
sudo nano /etc/systemd/system/fan-pwm.service
```

Paste:

```ini
[Unit]
Description=PWM Fan Controller
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/bin/fan_pwm.py
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```

Save and exit.

---

# 8. Enable Service

Reload systemd:

```bash
sudo systemctl daemon-reload
```

Enable startup:

```bash
sudo systemctl enable fan-pwm.service
```

Start immediately:

```bash
sudo systemctl start fan-pwm.service
```

---

# 9. Verify Service

Check status:

```bash
sudo systemctl status fan-pwm.service
```

Expected:

```text
Active: active (running)
```

---

# 10. View Logs

Show recent messages:

```bash
sudo journalctl -u fan-pwm.service -n 50
```

Follow live output:

```bash
sudo journalctl -u fan-pwm.service -f
```

---

# 11. Service Management

Stop:

```bash
sudo systemctl stop fan-pwm.service
```

Start:

```bash
sudo systemctl start fan-pwm.service
```

Restart:

```bash
sudo systemctl restart fan-pwm.service
```

Disable startup:

```bash
sudo systemctl disable fan-pwm.service
```

Enable startup:

```bash
sudo systemctl enable fan-pwm.service
```

---

# 12. Notes

Hardware:

* Board: Radxa Zero 3W
* SoC: RK3566
* Fan: 5V 2-wire brushless fan
* Control: MOSFET
* GPIO: `gpiochip3`, line `20` (PIN_7)

Tuning:

* Measured minimum working duty cycle: approximately 19%
* Configured minimum duty cycle: 25% for reliable startup
* PWM frequency: 25 Hz

The controller automatically ramps fan speed from 0% to 100% based on CPU temperature and starts on every boot.