#!/usr/bin/env python3
"""
MQTT Publisher Example
MQTT 발행자 예제

Publishes simulated sensor data to an MQTT broker.

Install:
    pip install paho-mqtt

Usage:
    python3 mqtt_publisher.py
    python3 mqtt_publisher.py --broker 192.168.1.100 --topic sensor/temp
"""

import paho.mqtt.client as mqtt
import json
import time
import random
import argparse
from datetime import datetime

# Default configuration
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "sensor/temperature"
DEFAULT_INTERVAL = 5
DEFAULT_SENSOR_ID = "temp_sensor_01"

class MQTTPublisher:
    """MQTT Sensor Data Publisher"""

    def __init__(self, broker: str, port: int, sensor_id: str):
        self.broker = broker
        self.port = port
        self.sensor_id = sensor_id

        # Create MQTT client
        self.client = mqtt.Client(client_id=f"publisher_{sensor_id}")
        self.client.on_connect = self._on_connect
        self.client.on_publish = self._on_publish
        self.client.on_disconnect = self._on_disconnect

        self.connected = False
        self.message_count = 0

    def _on_connect(self, client, userdata, flags, rc):
        """Connection callback"""
        if rc == 0:
            print(f"Connected to MQTT broker: {self.broker}:{self.port}")
            self.connected = True
        else:
            print(f"Connection failed with code: {rc}")

    def _on_publish(self, client, userdata, mid):
        """Publish callback"""
        self.message_count += 1
        print(f"  [Published] Message ID: {mid}, Total: {self.message_count}")

    def _on_disconnect(self, client, userdata, rc):
        """Disconnection callback"""
        print(f"Disconnected from broker (rc={rc})")
        self.connected = False

    def connect(self):
        """Connect to broker"""
        print(f"Connecting to {self.broker}:{self.port}...")
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

        # Wait for connection
        timeout = 5
        while not self.connected and timeout > 0:
            time.sleep(1)
            timeout -= 1

        if not self.connected:
            raise ConnectionError("Failed to connect to MQTT broker")

    def disconnect(self):
        """Disconnect from broker"""
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected from broker")

    def publish(self, topic: str, data: dict, qos: int = 1, retain: bool = False):
        """Publish message to topic"""
        payload = json.dumps(data)
        result = self.client.publish(topic, payload, qos=qos, retain=retain)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {topic}: {payload}")
            return True
        else:
            print(f"Publish failed with code: {result.rc}")
            return False

    def generate_sensor_data(self) -> dict:
        """Generate simulated sensor data"""
        return {
            "sensor_id": self.sensor_id,
            "temperature": round(20 + random.uniform(-5, 10), 1),
            "humidity": round(50 + random.uniform(-20, 20), 1),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MQTT Publisher")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT broker port")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Topic to publish to")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Publish interval (seconds)")
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID, help="Sensor ID")
    args = parser.parse_args()

    print("=== MQTT Publisher ===")
    print(f"Broker:   {args.broker}:{args.port}")
    print(f"Topic:    {args.topic}")
    print(f"Interval: {args.interval} seconds")
    print(f"Sensor:   {args.sensor_id}")
    print()

    publisher = MQTTPublisher(args.broker, args.port, args.sensor_id)

    try:
        publisher.connect()

        print("Publishing sensor data... (Ctrl+C to stop)")
        while True:
            data = publisher.generate_sensor_data()
            publisher.publish(args.topic, data, qos=1, retain=True)
            time.sleep(args.interval)

    except ConnectionError as e:
        print(f"Connection error: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        publisher.disconnect()
        print(f"Total messages published: {publisher.message_count}")

if __name__ == "__main__":
    main()
