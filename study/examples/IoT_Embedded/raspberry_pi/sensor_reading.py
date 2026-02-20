#!/usr/bin/env python3
"""
DHT11 Temperature and Humidity Sensor Reading
DHT11 온습도 센서 읽기 예제

Hardware:
- DHT11 VCC  -> 3.3V (pin 1)
- DHT11 DATA -> GPIO4 (pin 7) with 10k pull-up resistor
- DHT11 GND  -> GND (pin 6)

Install:
    pip install adafruit-circuitpython-dht
    sudo apt install libgpiod2

Usage:
    python3 sensor_reading.py
"""

import time
import json
from datetime import datetime

# Try to import DHT library
try:
    import adafruit_dht
    import board
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False
    print("Warning: adafruit_dht not available. Running in simulation mode.")

# Configuration
SENSOR_PIN = 4  # GPIO4
READ_INTERVAL = 5  # seconds

class SensorReader:
    """DHT11 Sensor Reader"""

    def __init__(self, pin: int = SENSOR_PIN):
        self.pin = pin
        self.dht = None

        if DHT_AVAILABLE:
            gpio_pin = getattr(board, f"D{pin}")
            self.dht = adafruit_dht.DHT11(gpio_pin)

    def read(self) -> dict:
        """Read temperature and humidity from sensor"""
        if not DHT_AVAILABLE or not self.dht:
            # Simulation mode
            import random
            return {
                "temperature": round(20 + random.uniform(0, 10), 1),
                "humidity": round(40 + random.uniform(0, 30), 1),
                "status": "simulated",
                "timestamp": datetime.now().isoformat()
            }

        try:
            temperature = self.dht.temperature
            humidity = self.dht.humidity

            if temperature is not None and humidity is not None:
                return {
                    "temperature": temperature,
                    "humidity": humidity,
                    "status": "ok",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to read sensor",
                    "timestamp": datetime.now().isoformat()
                }

        except RuntimeError as e:
            # DHT sensors occasionally fail to read
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def close(self):
        """Clean up sensor resources"""
        if self.dht:
            self.dht.exit()

def main():
    """Main function to read and display sensor data"""
    print("=== DHT11 Sensor Reader ===")
    print(f"Reading from GPIO{SENSOR_PIN}")
    print(f"Interval: {READ_INTERVAL} seconds")
    print("Press Ctrl+C to exit\n")

    sensor = SensorReader(SENSOR_PIN)

    try:
        while True:
            data = sensor.read()

            if data.get("status") == "ok" or data.get("status") == "simulated":
                print(f"Temperature: {data['temperature']:.1f}°C")
                print(f"Humidity:    {data['humidity']:.1f}%")
                print(f"Status:      {data['status']}")
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")

            print(f"Time:        {data['timestamp']}")
            print("-" * 40)

            # Save to JSON file (optional)
            with open("sensor_log.json", "a") as f:
                f.write(json.dumps(data) + "\n")

            time.sleep(READ_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        sensor.close()
        print("Sensor cleanup complete")

if __name__ == "__main__":
    main()
