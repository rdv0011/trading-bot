#!/usr/bin/env python3
"""
PWM Fan Controller — libgpiod v2, CPU-temperature-based speed curve.

Installed to /usr/local/bin/fan_pwm.py and managed by a systemd service.

See plans/update_fan_control.md for full setup instructions.
"""

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
