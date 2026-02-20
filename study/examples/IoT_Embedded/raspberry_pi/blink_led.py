#!/usr/bin/env python3
"""
LED Blink Example for Raspberry Pi
LED 깜빡이기 예제

Hardware:
- LED connected to GPIO17 (pin 11)
- 330 ohm resistor in series

Usage:
    python3 blink_led.py
"""

from gpiozero import LED
from time import sleep
import signal
import sys

# GPIO pin configuration
LED_PIN = 17

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nExiting...")
    sys.exit(0)

def main():
    """Main function to blink LED"""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize LED
    led = LED(LED_PIN)

    print(f"LED Blink started on GPIO{LED_PIN}")
    print("Press Ctrl+C to exit")

    # Blink loop
    blink_count = 0
    try:
        while True:
            led.on()
            print(f"LED ON  (count: {blink_count})")
            sleep(0.5)

            led.off()
            print(f"LED OFF (count: {blink_count})")
            sleep(0.5)

            blink_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        led.off()
        print("LED turned off, cleanup complete")

if __name__ == "__main__":
    main()
