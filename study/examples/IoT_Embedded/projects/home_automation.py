#!/usr/bin/env python3
"""
스마트홈 자동화 시스템 (Smart Home Automation System)

통합 홈 자동화 시스템으로 조명 제어, 환경 모니터링, MQTT 기반 장치 통신을 제공합니다.
시뮬레이션 모드를 지원하여 실제 하드웨어 없이도 동작 가능합니다.

주요 기능:
- 릴레이를 통한 조명/가전 제어 (시뮬레이션 모드 지원)
- 온습도 센서 모니터링 (시뮬레이션 데이터 생성)
- MQTT 기반 장치 통신 및 제어
- 자동화 규칙 엔진 (온도 기반, 모션 기반)
- 웹 대시보드 JSON API 제공
- 상태 관리 및 로깅

사용법:
    # 시뮬레이션 모드 (하드웨어 불필요)
    python home_automation.py --simulate

    # 실제 하드웨어 모드
    python home_automation.py

    # MQTT 브로커 지정
    python home_automation.py --broker mqtt.example.com --simulate
"""

import time
import json
import random
import threading
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from queue import Queue
import argparse

# 시뮬레이션 모드 플래그
SIMULATION_MODE = True

# 실제 하드웨어 라이브러리 (시뮬레이션 모드에서는 사용 안 함)
try:
    from gpiozero import OutputDevice
    import adafruit_dht
    import board
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("하드웨어 라이브러리 없음. 시뮬레이션 모드로 실행됩니다.")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("paho-mqtt 라이브러리가 없습니다. MQTT 기능이 비활성화됩니다.")


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class Light:
    """조명 장치 데이터 클래스"""
    id: str
    name: str
    gpio_pin: int
    location: str
    is_on: bool = False

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return asdict(self)


@dataclass
class SensorReading:
    """센서 데이터 클래스"""
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: datetime

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "sensor_id": self.sensor_id,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================
# 조명 제어기 (Light Controller)
# ============================================================

class LightController:
    """
    조명 제어 클래스

    릴레이를 통해 조명을 제어합니다.
    시뮬레이션 모드에서는 실제 GPIO 대신 상태만 관리합니다.
    """

    def __init__(self, config: dict, simulate: bool = True):
        """
        초기화

        Args:
            config: 조명 설정 (lights 리스트 포함)
            simulate: 시뮬레이션 모드 여부
        """
        self.simulate = simulate
        self.lights: Dict[str, Light] = {}
        self.relays: Dict[str, any] = {}

        # 조명 설정
        for light_config in config.get('lights', []):
            light = Light(**light_config)
            self.lights[light.id] = light

            # 릴레이 초기화
            if not simulate and HARDWARE_AVAILABLE:
                # 실제 하드웨어: gpiozero 사용
                relay = OutputDevice(
                    light.gpio_pin,
                    active_high=False,  # Active Low 릴레이
                    initial_value=False
                )
                self.relays[light.id] = relay
            else:
                # 시뮬레이션: None 저장
                self.relays[light.id] = None

        logging.info(f"LightController 초기화 완료 (시뮬레이션={simulate}, 조명={len(self.lights)}개)")

    def turn_on(self, light_id: str) -> bool:
        """
        조명 켜기

        Args:
            light_id: 조명 ID

        Returns:
            성공 여부
        """
        if light_id not in self.lights:
            logging.warning(f"조명 ID '{light_id}' 없음")
            return False

        if self.simulate:
            # 시뮬레이션: 상태만 변경
            self.lights[light_id].is_on = True
            logging.info(f"[시뮬] 조명 ON: {self.lights[light_id].name}")
        else:
            # 실제 하드웨어: 릴레이 제어
            self.relays[light_id].on()
            self.lights[light_id].is_on = True
            logging.info(f"조명 ON: {self.lights[light_id].name}")

        return True

    def turn_off(self, light_id: str) -> bool:
        """
        조명 끄기

        Args:
            light_id: 조명 ID

        Returns:
            성공 여부
        """
        if light_id not in self.lights:
            logging.warning(f"조명 ID '{light_id}' 없음")
            return False

        if self.simulate:
            # 시뮬레이션
            self.lights[light_id].is_on = False
            logging.info(f"[시뮬] 조명 OFF: {self.lights[light_id].name}")
        else:
            # 실제 하드웨어
            self.relays[light_id].off()
            self.lights[light_id].is_on = False
            logging.info(f"조명 OFF: {self.lights[light_id].name}")

        return True

    def toggle(self, light_id: str) -> bool:
        """
        조명 토글

        Args:
            light_id: 조명 ID

        Returns:
            성공 여부
        """
        if light_id not in self.lights:
            return False

        if self.lights[light_id].is_on:
            return self.turn_off(light_id)
        else:
            return self.turn_on(light_id)

    def get_status(self, light_id: str = None) -> Optional[dict]:
        """
        조명 상태 조회

        Args:
            light_id: 조명 ID (None이면 전체)

        Returns:
            상태 딕셔너리
        """
        if light_id:
            light = self.lights.get(light_id)
            if light:
                return light.to_dict()
            return None

        # 전체 조명 상태
        return {
            "lights": [light.to_dict() for light in self.lights.values()]
        }

    def all_off(self):
        """모든 조명 끄기"""
        for light_id in self.lights:
            self.turn_off(light_id)
        logging.info("모든 조명 OFF")

    def all_on(self):
        """모든 조명 켜기"""
        for light_id in self.lights:
            self.turn_on(light_id)
        logging.info("모든 조명 ON")

    def cleanup(self):
        """정리 (프로그램 종료 시 호출)"""
        if not self.simulate:
            for relay in self.relays.values():
                if relay:
                    relay.close()
        logging.info("LightController 정리 완료")


# ============================================================
# 환경 센서 모니터 (Environment Monitor)
# ============================================================

class EnvironmentMonitor:
    """
    환경 센서 모니터링 클래스

    DHT11 센서로 온도/습도를 주기적으로 읽고 히스토리를 저장합니다.
    시뮬레이션 모드에서는 랜덤 데이터를 생성합니다.
    """

    def __init__(self, sensor_pin: int = 4, sensor_id: str = "env_01", simulate: bool = True):
        """
        초기화

        Args:
            sensor_pin: GPIO 핀 번호
            sensor_id: 센서 ID
            simulate: 시뮬레이션 모드 여부
        """
        self.sensor_id = sensor_id
        self.sensor_pin = sensor_pin
        self.simulate = simulate

        # DHT 센서 초기화
        self.dht = None
        if not simulate and HARDWARE_AVAILABLE:
            try:
                self.dht = adafruit_dht.DHT11(getattr(board, f"D{sensor_pin}"))
            except Exception as e:
                logging.error(f"DHT 센서 초기화 실패: {e}")
                self.simulate = True

        # 데이터 큐 및 히스토리
        self.data_queue = Queue()
        self.latest_reading: Optional[SensorReading] = None
        self.readings_history: List[SensorReading] = []
        self.max_history = 1000

        # 스레드 제어
        self.running = False
        self.thread = None

        # 시뮬레이션용 현재 값
        self.sim_temperature = 25.0
        self.sim_humidity = 60.0

        logging.info(f"EnvironmentMonitor 초기화 (시뮬레이션={simulate})")

    def read_sensor(self) -> Optional[SensorReading]:
        """
        센서 읽기

        Returns:
            센서 데이터 또는 None (실패 시)
        """
        if self.simulate:
            # 시뮬레이션: 랜덤 변화 생성
            self.sim_temperature += random.uniform(-0.5, 0.5)
            self.sim_temperature = max(10, min(40, self.sim_temperature))

            self.sim_humidity += random.uniform(-2, 2)
            self.sim_humidity = max(30, min(90, self.sim_humidity))

            reading = SensorReading(
                sensor_id=self.sensor_id,
                temperature=round(self.sim_temperature, 1),
                humidity=round(self.sim_humidity, 1),
                timestamp=datetime.now()
            )
            return reading

        else:
            # 실제 센서
            try:
                temperature = self.dht.temperature
                humidity = self.dht.humidity

                if temperature is not None and humidity is not None:
                    reading = SensorReading(
                        sensor_id=self.sensor_id,
                        temperature=temperature,
                        humidity=humidity,
                        timestamp=datetime.now()
                    )
                    return reading

            except RuntimeError as e:
                # DHT 센서는 가끔 읽기 실패 (정상)
                logging.debug(f"센서 읽기 실패 (정상): {e}")
            except Exception as e:
                logging.error(f"센서 읽기 오류: {e}")

            return None

    def _monitor_loop(self, interval: int):
        """
        모니터링 루프 (백그라운드 스레드)

        Args:
            interval: 읽기 간격 (초)
        """
        while self.running:
            reading = self.read_sensor()

            if reading:
                # 최신 데이터 업데이트
                self.latest_reading = reading

                # 히스토리 저장
                self.readings_history.append(reading)
                if len(self.readings_history) > self.max_history:
                    self.readings_history.pop(0)

                # 큐에 추가 (외부 구독자용)
                self.data_queue.put(reading)

            time.sleep(interval)

    def start(self, interval: int = 5):
        """
        모니터링 시작

        Args:
            interval: 읽기 간격 (초)
        """
        if self.running:
            logging.warning("이미 모니터링 중입니다")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.thread.start()
        logging.info(f"환경 모니터링 시작 (간격={interval}초)")

    def stop(self):
        """모니터링 중지"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        if self.dht and not self.simulate:
            self.dht.exit()

        logging.info("환경 모니터링 중지")

    def get_latest(self) -> Optional[dict]:
        """최신 센서 데이터 반환"""
        if self.latest_reading:
            return self.latest_reading.to_dict()
        return None

    def get_stats(self) -> dict:
        """
        통계 데이터 반환

        Returns:
            최소/최대/평균 통계
        """
        if not self.readings_history:
            return {}

        temps = [r.temperature for r in self.readings_history]
        humids = [r.humidity for r in self.readings_history]

        return {
            "count": len(self.readings_history),
            "temperature": {
                "min": round(min(temps), 1),
                "max": round(max(temps), 1),
                "avg": round(sum(temps) / len(temps), 1)
            },
            "humidity": {
                "min": round(min(humids), 1),
                "max": round(max(humids), 1),
                "avg": round(sum(humids) / len(humids), 1)
            }
        }


# ============================================================
# MQTT 핸들러
# ============================================================

class SmartHomeMQTT:
    """
    MQTT 기반 스마트홈 제어 핸들러

    MQTT 브로커를 통해 장치를 제어하고 센서 데이터를 발행합니다.
    """

    TOPICS = {
        "light_command": "home/+/light/command",
        "light_status": "home/{}/light/status",
        "sensor_data": "home/sensor/{}",
        "motion": "home/motion/{}",
        "system": "home/system/status",
        "automation": "home/automation/event"
    }

    def __init__(self, light_controller: LightController,
                 env_monitor: EnvironmentMonitor,
                 broker: str = "localhost",
                 port: int = 1883):
        """
        초기화

        Args:
            light_controller: 조명 제어기
            env_monitor: 환경 모니터
            broker: MQTT 브로커 주소
            port: MQTT 포트
        """
        self.light_controller = light_controller
        self.env_monitor = env_monitor
        self.broker = broker
        self.port = port

        if not MQTT_AVAILABLE:
            logging.warning("MQTT 라이브러리 없음. MQTT 기능 비활성화")
            self.client = None
            return

        # MQTT 클라이언트 생성
        self.client = mqtt.Client(client_id="smart_home_gateway")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # Last Will and Testament (LWT) 설정
        self.client.will_set(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        logging.info(f"MQTT 클라이언트 생성 (브로커={broker}:{port})")

    def connect(self):
        """MQTT 브로커 연결"""
        if not self.client:
            return

        try:
            self.client.connect(self.broker, self.port)
            logging.info("MQTT 브로커 연결 시도 중...")
        except Exception as e:
            logging.error(f"MQTT 연결 실패: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT 연결 콜백"""
        if rc == 0:
            logging.info("MQTT 브로커 연결 성공")

            # 토픽 구독
            client.subscribe(self.TOPICS["light_command"])
            logging.info(f"토픽 구독: {self.TOPICS['light_command']}")

            # 온라인 상태 발행
            client.publish(
                self.TOPICS["system"],
                json.dumps({
                    "status": "online",
                    "timestamp": datetime.now().isoformat()
                }),
                qos=1,
                retain=True
            )
        else:
            logging.error(f"MQTT 연결 실패 (코드={rc})")

    def _on_message(self, client, userdata, msg):
        """MQTT 메시지 수신 콜백"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            logging.debug(f"MQTT 수신: {topic} = {payload}")

            # 조명 명령 처리
            if "light/command" in topic:
                self._handle_light_command(topic, payload)

        except json.JSONDecodeError:
            logging.error(f"잘못된 JSON: {msg.payload}")
        except Exception as e:
            logging.error(f"메시지 처리 오류: {e}")

    def _handle_light_command(self, topic: str, payload: dict):
        """
        조명 명령 처리

        Args:
            topic: MQTT 토픽 (home/{room}/light/command)
            payload: 명령 데이터 {"command": "on|off|toggle"}
        """
        # 토픽에서 방 ID 추출
        parts = topic.split('/')
        room = parts[1] if len(parts) >= 2 else None

        if not room:
            logging.warning(f"방 ID 없음: {topic}")
            return

        command = payload.get("command")

        # 명령 실행
        result = False
        if command == "on":
            result = self.light_controller.turn_on(room)
        elif command == "off":
            result = self.light_controller.turn_off(room)
        elif command == "toggle":
            result = self.light_controller.toggle(room)
        else:
            logging.warning(f"알 수 없는 명령: {command}")

        # 상태 발행
        if result:
            status = self.light_controller.get_status(room)
            if status:
                self.publish_light_status(room, status)

    def publish_light_status(self, room: str, status: dict):
        """
        조명 상태 발행

        Args:
            room: 방 ID
            status: 상태 데이터
        """
        if not self.client:
            return

        topic = self.TOPICS["light_status"].format(room)
        self.client.publish(topic, json.dumps(status), qos=1, retain=True)
        logging.debug(f"조명 상태 발행: {topic}")

    def publish_sensor_data(self, sensor_id: str, data: dict):
        """
        센서 데이터 발행

        Args:
            sensor_id: 센서 ID
            data: 센서 데이터
        """
        if not self.client:
            return

        topic = self.TOPICS["sensor_data"].format(sensor_id)
        self.client.publish(topic, json.dumps(data), qos=0)

    def publish_motion(self, sensor_id: str, detected: bool):
        """
        모션 감지 발행

        Args:
            sensor_id: 센서 ID
            detected: 감지 여부
        """
        if not self.client:
            return

        topic = self.TOPICS["motion"].format(sensor_id)
        data = {
            "detected": detected,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(topic, json.dumps(data), qos=1)

    def publish_automation_event(self, event_type: str, details: dict):
        """
        자동화 이벤트 발행

        Args:
            event_type: 이벤트 타입
            details: 상세 정보
        """
        if not self.client:
            return

        data = {
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(self.TOPICS["automation"], json.dumps(data), qos=1)

    def start(self):
        """MQTT 루프 시작"""
        if self.client:
            self.client.loop_start()

    def stop(self):
        """MQTT 중지"""
        if not self.client:
            return

        # 오프라인 상태 발행
        self.client.publish(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        self.client.loop_stop()
        self.client.disconnect()
        logging.info("MQTT 연결 종료")


# ============================================================
# 자동화 규칙 엔진
# ============================================================

class AutomationEngine:
    """
    자동화 규칙 엔진

    센서 데이터 기반으로 조명/가전을 자동 제어합니다.
    """

    def __init__(self, light_controller: LightController, mqtt_handler: SmartHomeMQTT):
        """
        초기화

        Args:
            light_controller: 조명 제어기
            mqtt_handler: MQTT 핸들러
        """
        self.light_controller = light_controller
        self.mqtt_handler = mqtt_handler
        self.rules: List[dict] = []

        logging.info("AutomationEngine 초기화")

    def add_rule(self, name: str, condition: Callable, action: Callable):
        """
        규칙 추가

        Args:
            name: 규칙 이름
            condition: 조건 함수 (True/False 반환)
            action: 실행 함수
        """
        self.rules.append({
            "name": name,
            "condition": condition,
            "action": action,
            "last_triggered": None
        })
        logging.info(f"자동화 규칙 추가: {name}")

    def check_rules(self, sensor_data: dict):
        """
        규칙 검사 및 실행

        Args:
            sensor_data: 센서 데이터
        """
        for rule in self.rules:
            try:
                if rule["condition"](sensor_data):
                    # 조건 만족 시 액션 실행
                    logging.info(f"자동화 규칙 트리거: {rule['name']}")
                    rule["action"](sensor_data)
                    rule["last_triggered"] = datetime.now()

                    # MQTT 이벤트 발행
                    self.mqtt_handler.publish_automation_event(
                        event_type=rule["name"],
                        details=sensor_data
                    )

            except Exception as e:
                logging.error(f"규칙 실행 오류 ({rule['name']}): {e}")

    def get_rules_status(self) -> List[dict]:
        """규칙 상태 조회"""
        return [
            {
                "name": rule["name"],
                "last_triggered": rule["last_triggered"].isoformat() if rule["last_triggered"] else None
            }
            for rule in self.rules
        ]


# ============================================================
# 스마트홈 게이트웨이 (통합 시스템)
# ============================================================

class SmartHomeGateway:
    """
    스마트홈 통합 게이트웨이

    모든 컴포넌트를 통합하여 스마트홈 시스템을 운영합니다.
    """

    def __init__(self, config: dict, simulate: bool = True):
        """
        초기화

        Args:
            config: 설정 딕셔너리
            simulate: 시뮬레이션 모드 여부
        """
        self.config = config
        self.simulate = simulate

        # 조명 제어기
        self.light_controller = LightController(config, simulate=simulate)

        # 환경 모니터
        self.env_monitor = EnvironmentMonitor(
            sensor_pin=config.get('dht_pin', 4),
            sensor_id="env_01",
            simulate=simulate
        )

        # MQTT 핸들러
        self.mqtt_handler = SmartHomeMQTT(
            self.light_controller,
            self.env_monitor,
            broker=config.get('mqtt_broker', 'localhost'),
            port=config.get('mqtt_port', 1883)
        )

        # 자동화 엔진
        self.automation_engine = AutomationEngine(
            self.light_controller,
            self.mqtt_handler
        )

        # 스레드 제어
        self.running = False
        self.threads = []

        # 자동화 규칙 설정
        self._setup_automation_rules()

        logging.info("SmartHomeGateway 초기화 완료")

    def _setup_automation_rules(self):
        """자동화 규칙 설정"""

        # 규칙 1: 온도가 30도 이상이면 거실 조명 켜기 (예: 에어컨 대신)
        def temp_high_condition(data):
            return data.get("temperature", 0) > 30

        def temp_high_action(data):
            self.light_controller.turn_on("living_room")
            logging.info(f"[자동화] 고온 감지 ({data['temperature']}°C) - 거실 조명 ON")

        self.automation_engine.add_rule(
            "high_temperature_alert",
            temp_high_condition,
            temp_high_action
        )

        # 규칙 2: 온도가 20도 이하이면 모든 조명 끄기
        def temp_low_condition(data):
            return data.get("temperature", 100) < 20

        def temp_low_action(data):
            self.light_controller.all_off()
            logging.info(f"[자동화] 저온 감지 ({data['temperature']}°C) - 모든 조명 OFF")

        self.automation_engine.add_rule(
            "low_temperature_save",
            temp_low_condition,
            temp_low_action
        )

        # 규칙 3: 습도가 80% 이상이면 욕실 조명 켜기
        def humidity_high_condition(data):
            return data.get("humidity", 0) > 80

        def humidity_high_action(data):
            self.light_controller.turn_on("bathroom")
            logging.info(f"[자동화] 고습도 감지 ({data['humidity']}%) - 욕실 조명 ON")

        self.automation_engine.add_rule(
            "high_humidity_ventilation",
            humidity_high_condition,
            humidity_high_action
        )

    def _sensor_publish_loop(self, interval: int):
        """
        센서 데이터 발행 루프

        Args:
            interval: 발행 간격 (초)
        """
        while self.running:
            data = self.env_monitor.get_latest()
            if data:
                # MQTT 발행
                self.mqtt_handler.publish_sensor_data("env_01", data)

                # 자동화 규칙 검사
                self.automation_engine.check_rules(data)

            time.sleep(interval)

    def _status_report_loop(self, interval: int):
        """
        상태 리포트 루프

        Args:
            interval: 리포트 간격 (초)
        """
        while self.running:
            # 시스템 상태 출력
            sensor_data = self.env_monitor.get_latest()
            light_status = self.light_controller.get_status()

            logging.info("=" * 60)
            logging.info("시스템 상태 리포트")
            logging.info("-" * 60)

            if sensor_data:
                logging.info(f"온도: {sensor_data['temperature']}°C, 습도: {sensor_data['humidity']}%")

            if light_status:
                for light in light_status["lights"]:
                    status = "ON" if light["is_on"] else "OFF"
                    logging.info(f"{light['name']} ({light['location']}): {status}")

            logging.info("=" * 60)

            time.sleep(interval)

    def start(self):
        """게이트웨이 시작"""
        if self.running:
            logging.warning("이미 실행 중입니다")
            return

        logging.info("=" * 60)
        logging.info("스마트홈 게이트웨이 시작")
        logging.info(f"시뮬레이션 모드: {self.simulate}")
        logging.info("=" * 60)

        self.running = True

        # 환경 모니터링 시작
        self.env_monitor.start(interval=5)

        # MQTT 연결 및 시작
        self.mqtt_handler.connect()
        self.mqtt_handler.start()

        # 센서 데이터 발행 스레드
        sensor_thread = threading.Thread(
            target=self._sensor_publish_loop,
            args=(10,),
            daemon=True
        )
        sensor_thread.start()
        self.threads.append(sensor_thread)

        # 상태 리포트 스레드
        status_thread = threading.Thread(
            target=self._status_report_loop,
            args=(30,),
            daemon=True
        )
        status_thread.start()
        self.threads.append(status_thread)

        logging.info("게이트웨이 실행 중...")

    def stop(self):
        """게이트웨이 중지"""
        if not self.running:
            return

        logging.info("게이트웨이 중지 중...")

        self.running = False

        # 컴포넌트 정리
        self.env_monitor.stop()
        self.mqtt_handler.stop()
        self.light_controller.all_off()
        self.light_controller.cleanup()

        # 스레드 종료 대기
        for thread in self.threads:
            thread.join(timeout=2)

        logging.info("게이트웨이 중지 완료")

    def run(self):
        """메인 실행 루프"""
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nKeyboardInterrupt 수신")
        finally:
            self.stop()

    def get_dashboard_data(self) -> dict:
        """
        웹 대시보드용 JSON 데이터 제공

        Returns:
            시스템 전체 상태
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "lights": self.light_controller.get_status(),
            "sensor": {
                "latest": self.env_monitor.get_latest(),
                "stats": self.env_monitor.get_stats()
            },
            "automation": {
                "rules": self.automation_engine.get_rules_status()
            },
            "system": {
                "running": self.running,
                "simulation_mode": self.simulate
            }
        }


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 함수"""

    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description="스마트홈 자동화 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python home_automation.py --simulate
  python home_automation.py --broker mqtt.example.com --simulate
  python home_automation.py --loglevel DEBUG --simulate
        """
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="시뮬레이션 모드 (하드웨어 불필요)"
    )

    parser.add_argument(
        "--broker",
        type=str,
        default="localhost",
        help="MQTT 브로커 주소 (기본값: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT 포트 (기본값: 1883)"
    )

    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본값: INFO)"
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 시뮬레이션 모드 설정
    simulate = args.simulate or not HARDWARE_AVAILABLE

    if simulate:
        logging.info("시뮬레이션 모드로 실행합니다 (하드웨어 불필요)")
    else:
        logging.info("실제 하드웨어 모드로 실행합니다")

    # 설정
    config = {
        "lights": [
            {
                "id": "living_room",
                "name": "거실 조명",
                "gpio_pin": 17,
                "location": "거실"
            },
            {
                "id": "bedroom",
                "name": "침실 조명",
                "gpio_pin": 27,
                "location": "침실"
            },
            {
                "id": "kitchen",
                "name": "주방 조명",
                "gpio_pin": 22,
                "location": "주방"
            },
            {
                "id": "bathroom",
                "name": "욕실 조명",
                "gpio_pin": 23,
                "location": "욕실"
            }
        ],
        "dht_pin": 4,
        "mqtt_broker": args.broker,
        "mqtt_port": args.port
    }

    # 게이트웨이 생성 및 실행
    gateway = SmartHomeGateway(config, simulate=simulate)

    # 데모: 5초 후 대시보드 데이터 출력
    def print_dashboard():
        time.sleep(5)
        dashboard_data = gateway.get_dashboard_data()
        logging.info("\n" + "=" * 60)
        logging.info("대시보드 데이터 (JSON API)")
        logging.info("=" * 60)
        print(json.dumps(dashboard_data, indent=2, ensure_ascii=False))
        logging.info("=" * 60)

    demo_thread = threading.Thread(target=print_dashboard, daemon=True)
    demo_thread.start()

    # 메인 루프 실행
    gateway.run()


if __name__ == "__main__":
    main()
