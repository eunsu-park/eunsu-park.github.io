#!/usr/bin/env python3
"""
MQTT Subscriber Example
MQTT 구독자 예제

Subscribes to MQTT topics and receives sensor data.

Install:
    pip install paho-mqtt

Usage:
    python3 mqtt_subscriber.py
    python3 mqtt_subscriber.py --broker 192.168.1.100 --topic "sensor/#"
"""

import paho.mqtt.client as mqtt
import json
import argparse
from datetime import datetime

# Default configuration
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883
DEFAULT_TOPICS = ["sensor/#", "device/+/status"]

class MQTTSubscriber:
    """MQTT Message Subscriber"""

    def __init__(self, broker: str, port: int, topics: list):
        self.broker = broker
        self.port = port
        self.topics = topics

        # Create MQTT client
        self.client = mqtt.Client(client_id="subscriber_monitor")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.connected = False
        self.message_count = 0
        self.message_handlers = {}

    def _on_connect(self, client, userdata, flags, rc):
        """Connection callback"""
        if rc == 0:
            print(f"Connected to MQTT broker: {self.broker}:{self.port}")
            self.connected = True

            # Subscribe to topics
            for topic in self.topics:
                client.subscribe(topic, qos=1)
                print(f"Subscribed to: {topic}")
        else:
            print(f"Connection failed with code: {rc}")

    def _on_message(self, client, userdata, msg):
        """Message received callback"""
        self.message_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        try:
            # Try to parse as JSON
            payload = json.loads(msg.payload.decode('utf-8'))
            self._display_json_message(timestamp, msg.topic, payload)
        except json.JSONDecodeError:
            # Display as plain text
            payload = msg.payload.decode('utf-8')
            self._display_text_message(timestamp, msg.topic, payload)

        # Call custom handler if registered
        for pattern, handler in self.message_handlers.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                handler(msg.topic, payload)

    def _display_json_message(self, timestamp: str, topic: str, payload: dict):
        """Display formatted JSON message"""
        print(f"\n[{timestamp}] #{self.message_count}")
        print(f"Topic: {topic}")
        print(f"Data:")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print("-" * 50)

    def _display_text_message(self, timestamp: str, topic: str, payload: str):
        """Display plain text message"""
        print(f"\n[{timestamp}] #{self.message_count}")
        print(f"Topic: {topic}")
        print(f"Message: {payload}")
        print("-" * 50)

    def _on_disconnect(self, client, userdata, rc):
        """Disconnection callback"""
        print(f"Disconnected from broker (rc={rc})")
        self.connected = False

    def add_handler(self, topic_pattern: str, handler):
        """Add custom message handler for topic pattern"""
        self.message_handlers[topic_pattern] = handler

    def connect(self):
        """Connect to broker"""
        print(f"Connecting to {self.broker}:{self.port}...")
        self.client.connect(self.broker, self.port, keepalive=60)

    def run(self):
        """Run subscriber loop"""
        self.connect()
        print("\nWaiting for messages... (Ctrl+C to stop)")
        print("=" * 50)

        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.client.disconnect()
            print(f"Total messages received: {self.message_count}")

# Example custom handlers
def temperature_alert_handler(topic: str, payload: dict):
    """Alert when temperature exceeds threshold"""
    if isinstance(payload, dict):
        temp = payload.get("temperature")
        if temp and temp > 30:
            print(f"  [ALERT] High temperature: {temp}°C")

def motion_handler(topic: str, payload: dict):
    """Handle motion detection events"""
    if isinstance(payload, dict):
        if payload.get("motion_detected"):
            print(f"  [MOTION] Motion detected at {topic}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MQTT Subscriber")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT broker port")
    parser.add_argument("--topic", nargs="+", default=DEFAULT_TOPICS, help="Topics to subscribe")
    args = parser.parse_args()

    print("=== MQTT Subscriber ===")
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topics: {', '.join(args.topic)}")
    print()

    subscriber = MQTTSubscriber(args.broker, args.port, args.topic)

    # Add custom handlers
    subscriber.add_handler("sensor/+/temperature", temperature_alert_handler)
    subscriber.add_handler("sensor/+/motion", motion_handler)

    subscriber.run()

if __name__ == "__main__":
    main()
