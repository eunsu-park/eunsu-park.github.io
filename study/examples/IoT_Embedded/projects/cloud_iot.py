#!/usr/bin/env python3
"""
클라우드 IoT 통합 - 시뮬레이션 모드
AWS IoT Core 및 GCP Pub/Sub 연동 시뮬레이션

실제 클라우드 계정 및 자격 증명 없이 동작
"""

import json
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import random


# ==============================================================================
# 데이터 모델
# ==============================================================================

class CloudProvider(Enum):
    """클라우드 제공자"""
    AWS_IOT = "aws_iot"
    GCP_PUBSUB = "gcp_pubsub"
    SIMULATION = "simulation"


class MessageQoS(Enum):
    """MQTT QoS 레벨"""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


@dataclass
class IoTMessage:
    """IoT 메시지"""
    topic: str
    payload: Dict
    timestamp: datetime
    message_id: str
    qos: MessageQoS = MessageQoS.AT_LEAST_ONCE

    def to_json(self) -> str:
        """JSON 직렬화"""
        data = {
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "qos": self.qos.value
        }
        return json.dumps(data)


@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device_id: str
    device_type: str
    location: str
    firmware_version: str
    registered_at: datetime


@dataclass
class TelemetryData:
    """텔레메트리 데이터"""
    device_id: str
    temperature: float
    humidity: float
    pressure: float
    battery_level: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "device_id": self.device_id,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "battery_level": self.battery_level,
            "timestamp": self.timestamp.isoformat()
        }


# ==============================================================================
# AWS IoT Core 시뮬레이션
# ==============================================================================

class SimulatedAWSIoTClient:
    """AWS IoT Core 클라이언트 (시뮬레이션)"""

    def __init__(self, endpoint: str, cert_path: str, key_path: str,
                 ca_path: str, client_id: str):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.client_id = client_id
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}
        self.message_queue = queue.Queue()

        print(f"[AWS IoT 시뮬레이션] 클라이언트 생성")
        print(f"  - 엔드포인트: {endpoint}")
        print(f"  - 클라이언트 ID: {client_id}")
        print(f"  - 인증서: {cert_path}")

    def connect(self):
        """연결 (시뮬레이션)"""
        print(f"\n[AWS IoT 시뮬레이션] 연결 중...")
        time.sleep(0.5)  # 연결 지연 시뮬레이션

        # 인증서 검증 시뮬레이션
        print("  - TLS 핸드셰이크...")
        time.sleep(0.2)
        print("  - 인증서 검증...")
        time.sleep(0.2)
        print("  - MQTT 연결...")
        time.sleep(0.2)

        self.connected = True
        print("✓ 연결 성공!\n")

    def disconnect(self):
        """연결 해제"""
        if self.connected:
            print("[AWS IoT 시뮬레이션] 연결 해제")
            self.connected = False

    def publish(self, topic: str, payload: Dict, qos: MessageQoS = MessageQoS.AT_LEAST_ONCE):
        """메시지 발행"""
        if not self.connected:
            raise RuntimeError("연결되지 않음")

        message = IoTMessage(
            topic=topic,
            payload=payload,
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
            qos=qos
        )

        # 발행 시뮬레이션
        print(f"[AWS IoT 발행] {topic}")
        print(f"  메시지 ID: {message.message_id[:8]}...")
        print(f"  QoS: {qos.name}")
        print(f"  페이로드: {json.dumps(payload, indent=2)}")

        # 네트워크 지연 시뮬레이션
        time.sleep(random.uniform(0.01, 0.05))

        return message.message_id

    def subscribe(self, topic: str, callback: Callable):
        """토픽 구독"""
        if not self.connected:
            raise RuntimeError("연결되지 않음")

        self.subscriptions[topic] = callback
        print(f"[AWS IoT 구독] {topic}")

    def _simulate_incoming_message(self, topic: str, payload: Dict):
        """수신 메시지 시뮬레이션 (내부 테스트용)"""
        if topic in self.subscriptions:
            callback = self.subscriptions[topic]
            callback(topic, payload)


class AWSIoTDeviceManager:
    """AWS IoT 디바이스 관리자"""

    def __init__(self, client: SimulatedAWSIoTClient):
        self.client = client
        self.device_info: Optional[DeviceInfo] = None

    def register_device(self, device_info: DeviceInfo):
        """디바이스 등록"""
        self.device_info = device_info

        print("\n[AWS IoT] 디바이스 등록")
        print(f"  - ID: {device_info.device_id}")
        print(f"  - 타입: {device_info.device_type}")
        print(f"  - 위치: {device_info.location}")
        print(f"  - 펌웨어: {device_info.firmware_version}")

        # Thing 생성 시뮬레이션
        print("  - Thing 생성 중...")
        time.sleep(0.3)

        # 인증서 연결 시뮬레이션
        print("  - 인증서 연결 중...")
        time.sleep(0.3)

        # 정책 연결 시뮬레이션
        print("  - IoT 정책 연결 중...")
        time.sleep(0.3)

        print("✓ 디바이스 등록 완료\n")

    def publish_telemetry(self, data: TelemetryData):
        """텔레메트리 발행"""
        topic = f"device/{data.device_id}/telemetry"
        payload = data.to_dict()

        self.client.publish(topic, payload)

    def update_device_shadow(self, state: Dict):
        """Device Shadow 업데이트"""
        if not self.device_info:
            raise ValueError("디바이스 정보 없음")

        topic = f"$aws/things/{self.device_info.device_id}/shadow/update"

        shadow_payload = {
            "state": {
                "reported": state
            },
            "metadata": {
                "reported": {
                    k: {"timestamp": int(time.time())}
                    for k in state.keys()
                }
            }
        }

        print(f"\n[AWS IoT] Device Shadow 업데이트")
        print(f"  상태: {json.dumps(state, indent=2)}")

        self.client.publish(topic, shadow_payload)


# ==============================================================================
# GCP Pub/Sub 시뮬레이션
# ==============================================================================

class SimulatedGCPPubSubPublisher:
    """GCP Pub/Sub Publisher (시뮬레이션)"""

    def __init__(self, project_id: str, topic_id: str):
        self.project_id = project_id
        self.topic_id = topic_id
        self.topic_path = f"projects/{project_id}/topics/{topic_id}"

        print(f"[GCP Pub/Sub 시뮬레이션] Publisher 생성")
        print(f"  - 프로젝트: {project_id}")
        print(f"  - 토픽: {topic_id}")
        print(f"  - 경로: {self.topic_path}")

    def publish(self, data: Dict, **attributes) -> str:
        """메시지 발행"""
        message_id = str(uuid.uuid4())

        print(f"\n[GCP Pub/Sub 발행]")
        print(f"  토픽: {self.topic_id}")
        print(f"  메시지 ID: {message_id[:8]}...")

        if attributes:
            print(f"  속성: {attributes}")

        print(f"  데이터: {json.dumps(data, indent=2)}")

        # 네트워크 지연 시뮬레이션
        time.sleep(random.uniform(0.01, 0.05))

        print(f"✓ 발행 완료\n")
        return message_id

    def publish_batch(self, messages: List[Dict]) -> List[str]:
        """배치 발행"""
        print(f"\n[GCP Pub/Sub 배치 발행] {len(messages)}개 메시지")

        message_ids = []
        for i, data in enumerate(messages):
            message_id = str(uuid.uuid4())
            message_ids.append(message_id)
            print(f"  [{i+1}] {message_id[:8]}...")

        time.sleep(random.uniform(0.05, 0.1))
        print(f"✓ 배치 발행 완료\n")

        return message_ids


class SimulatedGCPPubSubSubscriber:
    """GCP Pub/Sub Subscriber (시뮬레이션)"""

    def __init__(self, project_id: str, subscription_id: str):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"
        self.message_queue = queue.Queue()

        print(f"[GCP Pub/Sub 시뮬레이션] Subscriber 생성")
        print(f"  - 구독: {subscription_id}")
        print(f"  - 경로: {self.subscription_path}")

    def pull(self, max_messages: int = 10) -> List[Dict]:
        """메시지 풀 (동기)"""
        print(f"\n[GCP Pub/Sub Pull] 최대 {max_messages}개 메시지")

        messages = []

        # 시뮬레이션: 큐에서 메시지 가져오기
        for _ in range(min(max_messages, self.message_queue.qsize())):
            try:
                msg = self.message_queue.get_nowait()
                messages.append(msg)
            except queue.Empty:
                break

        print(f"  수신: {len(messages)}개 메시지")

        # ACK 시뮬레이션
        if messages:
            print(f"  ACK 전송 중...")
            time.sleep(0.1)

        return messages

    def subscribe(self, callback: Callable):
        """스트리밍 구독 (비동기)"""
        print(f"\n[GCP Pub/Sub] 스트리밍 구독 시작")

        def streaming_loop():
            while True:
                try:
                    msg = self.message_queue.get(timeout=1)
                    callback(msg, {})
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"구독 오류: {e}")
                    break

        thread = threading.Thread(target=streaming_loop, daemon=True)
        thread.start()

        return thread


# ==============================================================================
# MQTT 메시지 포맷
# ==============================================================================

class MQTTMessageFormat:
    """MQTT 메시지 포맷 표준"""

    @staticmethod
    def create_telemetry(device_id: str, sensor_data: Dict) -> Dict:
        """텔레메트리 메시지 생성"""
        return {
            "device_id": device_id,
            "message_type": "telemetry",
            "data": sensor_data,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }

    @staticmethod
    def create_event(device_id: str, event_type: str, event_data: Dict) -> Dict:
        """이벤트 메시지 생성"""
        return {
            "device_id": device_id,
            "message_type": "event",
            "event_type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }

    @staticmethod
    def create_command(device_id: str, command: str, parameters: Dict) -> Dict:
        """명령 메시지 생성"""
        return {
            "device_id": device_id,
            "message_type": "command",
            "command": command,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "command_id": str(uuid.uuid4()),
            "version": "1.0"
        }

    @staticmethod
    def create_response(device_id: str, command_id: str, status: str, result: Dict) -> Dict:
        """명령 응답 메시지 생성"""
        return {
            "device_id": device_id,
            "message_type": "response",
            "command_id": command_id,
            "status": status,  # "success", "error", "timeout"
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }


# ==============================================================================
# 디바이스 프로비저닝
# ==============================================================================

class DeviceProvisioning:
    """디바이스 프로비저닝 (시뮬레이션)"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.provisioned_devices: Dict[str, DeviceInfo] = {}

        print(f"\n[프로비저닝] 초기화 ({provider.value})")

    def provision_device(self, device_id: str, device_type: str, location: str) -> DeviceInfo:
        """디바이스 프로비저닝"""
        print(f"\n[프로비저닝] 디바이스 등록 시작")
        print(f"  - ID: {device_id}")
        print(f"  - 타입: {device_type}")
        print(f"  - 위치: {location}")

        # 단계 1: 디바이스 생성
        print("\n  [1/4] 디바이스 생성 중...")
        time.sleep(0.3)

        # 단계 2: 인증서 생성
        print("  [2/4] 인증서 생성 중...")
        cert_arn = f"arn:aws:iot:region:account:cert/{uuid.uuid4()}"
        print(f"    인증서 ARN: {cert_arn}")
        time.sleep(0.3)

        # 단계 3: 정책 연결
        print("  [3/4] 정책 연결 중...")
        print("    정책: IoTDevicePolicy")
        time.sleep(0.3)

        # 단계 4: Thing 생성 및 연결
        print("  [4/4] Thing 생성 및 연결 중...")
        time.sleep(0.3)

        device_info = DeviceInfo(
            device_id=device_id,
            device_type=device_type,
            location=location,
            firmware_version="1.0.0",
            registered_at=datetime.now()
        )

        self.provisioned_devices[device_id] = device_info

        print("\n✓ 프로비저닝 완료!")
        print(f"  인증서가 생성되었습니다: certs/{device_id}.cert.pem")
        print(f"  개인키가 생성되었습니다: certs/{device_id}.private.key\n")

        return device_info

    def deprovision_device(self, device_id: str):
        """디바이스 해제"""
        if device_id not in self.provisioned_devices:
            raise ValueError(f"디바이스 없음: {device_id}")

        print(f"\n[프로비저닝] 디바이스 해제: {device_id}")

        # 인증서 비활성화
        print("  - 인증서 비활성화 중...")
        time.sleep(0.2)

        # Thing 삭제
        print("  - Thing 삭제 중...")
        time.sleep(0.2)

        del self.provisioned_devices[device_id]
        print("✓ 해제 완료\n")


# ==============================================================================
# 통합 IoT 클라이언트
# ==============================================================================

class CloudIoTClient:
    """통합 클라우드 IoT 클라이언트"""

    def __init__(self, provider: CloudProvider, config: Dict):
        self.provider = provider
        self.config = config

        print("\n" + "="*60)
        print(f"클라우드 IoT 클라이언트 초기화 ({provider.value})")
        print("="*60)

        if provider == CloudProvider.AWS_IOT:
            self.client = SimulatedAWSIoTClient(
                endpoint=config.get('endpoint', 'simulated-endpoint.iot.region.amazonaws.com'),
                cert_path=config.get('cert_path', 'certs/device.cert.pem'),
                key_path=config.get('key_path', 'certs/device.private.key'),
                ca_path=config.get('ca_path', 'certs/root-CA.crt'),
                client_id=config.get('client_id', 'device-001')
            )

        elif provider == CloudProvider.GCP_PUBSUB:
            self.publisher = SimulatedGCPPubSubPublisher(
                project_id=config.get('project_id', 'my-iot-project'),
                topic_id=config.get('topic_id', 'iot-telemetry')
            )
            self.subscriber = SimulatedGCPPubSubSubscriber(
                project_id=config.get('project_id', 'my-iot-project'),
                subscription_id=config.get('subscription_id', 'iot-telemetry-sub')
            )

        self.message_stats = {
            "published": 0,
            "received": 0,
            "errors": 0
        }

    def connect(self):
        """연결"""
        if self.provider == CloudProvider.AWS_IOT:
            self.client.connect()

    def disconnect(self):
        """연결 해제"""
        if self.provider == CloudProvider.AWS_IOT:
            self.client.disconnect()

    def publish_telemetry(self, device_id: str, sensor_data: Dict):
        """텔레메트리 발행"""
        message = MQTTMessageFormat.create_telemetry(device_id, sensor_data)

        if self.provider == CloudProvider.AWS_IOT:
            topic = f"device/{device_id}/telemetry"
            self.client.publish(topic, message)
        elif self.provider == CloudProvider.GCP_PUBSUB:
            self.publisher.publish(message, device_id=device_id, message_type="telemetry")

        self.message_stats["published"] += 1

    def subscribe_commands(self, device_id: str, callback: Callable):
        """명령 구독"""
        def command_handler(topic: str, payload: Dict):
            print(f"\n[명령 수신] {topic}")
            print(f"  명령: {payload.get('command')}")
            print(f"  파라미터: {payload.get('parameters')}")

            # 콜백 실행
            callback(payload)

            # 응답 전송
            response = MQTTMessageFormat.create_response(
                device_id=device_id,
                command_id=payload.get('command_id'),
                status="success",
                result={"executed": True}
            )

            response_topic = f"device/{device_id}/response"
            self.client.publish(response_topic, response)

        if self.provider == CloudProvider.AWS_IOT:
            topic = f"device/{device_id}/command"
            self.client.subscribe(topic, command_handler)

    def get_statistics(self) -> Dict:
        """통계 조회"""
        return self.message_stats.copy()


# ==============================================================================
# 센서 데이터 시뮬레이터
# ==============================================================================

class SensorSimulator:
    """센서 데이터 시뮬레이터"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.base_temp = 25.0
        self.base_humidity = 60.0
        self.base_pressure = 1013.25
        self.battery_level = 100.0

    def generate_reading(self) -> TelemetryData:
        """센서 읽기 생성"""
        # 랜덤 변동 추가
        temp = self.base_temp + random.uniform(-2, 2)
        humidity = self.base_humidity + random.uniform(-5, 5)
        pressure = self.base_pressure + random.uniform(-2, 2)

        # 배터리 소모
        self.battery_level = max(0, self.battery_level - random.uniform(0.01, 0.05))

        return TelemetryData(
            device_id=self.device_id,
            temperature=round(temp, 2),
            humidity=round(humidity, 2),
            pressure=round(pressure, 2),
            battery_level=round(self.battery_level, 2),
            timestamp=datetime.now()
        )


# ==============================================================================
# 데모 시나리오
# ==============================================================================

def demo_aws_iot():
    """AWS IoT Core 데모"""
    print("\n" + "="*60)
    print("AWS IoT Core 데모")
    print("="*60)

    # 프로비저닝
    provisioning = DeviceProvisioning(CloudProvider.AWS_IOT)
    device_info = provisioning.provision_device(
        device_id="raspberry-pi-001",
        device_type="sensor-hub",
        location="Seoul, Korea"
    )

    # 클라이언트 생성
    config = {
        'endpoint': 'a1b2c3d4e5f6g7.iot.ap-northeast-2.amazonaws.com',
        'cert_path': f'certs/{device_info.device_id}.cert.pem',
        'key_path': f'certs/{device_info.device_id}.private.key',
        'ca_path': 'certs/AmazonRootCA1.pem',
        'client_id': device_info.device_id
    }

    client = CloudIoTClient(CloudProvider.AWS_IOT, config)
    client.connect()

    # 디바이스 관리자
    device_manager = AWSIoTDeviceManager(client.client)
    device_manager.register_device(device_info)

    # 센서 시뮬레이터
    sensor = SensorSimulator(device_info.device_id)

    # 명령 구독
    def on_command(payload: Dict):
        command = payload.get('command')
        print(f"\n명령 실행: {command}")

    client.subscribe_commands(device_info.device_id, on_command)

    # 텔레메트리 발행
    print("\n" + "-"*60)
    print("텔레메트리 발행 시작")
    print("-"*60)

    for i in range(5):
        print(f"\n[{i+1}/5] 센서 데이터 발행")

        # 센서 읽기
        data = sensor.generate_reading()
        print(f"  온도: {data.temperature}°C")
        print(f"  습도: {data.humidity}%")
        print(f"  압력: {data.pressure} hPa")
        print(f"  배터리: {data.battery_level}%")

        # 발행
        device_manager.publish_telemetry(data)

        # Device Shadow 업데이트
        if i % 2 == 0:
            shadow_state = {
                "temperature": data.temperature,
                "humidity": data.humidity,
                "battery": data.battery_level
            }
            device_manager.update_device_shadow(shadow_state)

        time.sleep(2)

    # 통계
    print("\n" + "="*60)
    print("AWS IoT 데모 완료")
    print("="*60)
    stats = client.get_statistics()
    print(f"발행된 메시지: {stats['published']}개")

    client.disconnect()


def demo_gcp_pubsub():
    """GCP Pub/Sub 데모"""
    print("\n" + "="*60)
    print("GCP Pub/Sub 데모")
    print("="*60)

    # 클라이언트 생성
    config = {
        'project_id': 'my-iot-project-123456',
        'topic_id': 'iot-telemetry',
        'subscription_id': 'iot-telemetry-sub'
    }

    client = CloudIoTClient(CloudProvider.GCP_PUBSUB, config)

    # 센서 시뮬레이터
    sensor = SensorSimulator("gcp-device-001")

    # 텔레메트리 발행
    print("\n" + "-"*60)
    print("텔레메트리 발행 시작")
    print("-"*60)

    messages = []
    for i in range(5):
        data = sensor.generate_reading()
        print(f"\n[{i+1}/5] 센서 데이터 생성")
        print(f"  온도: {data.temperature}°C")
        print(f"  습도: {data.humidity}%")

        # 개별 발행
        sensor_data = {
            "temperature": data.temperature,
            "humidity": data.humidity,
            "pressure": data.pressure,
            "battery_level": data.battery_level
        }

        client.publish_telemetry("gcp-device-001", sensor_data)
        messages.append(data.to_dict())

        time.sleep(1)

    # 배치 발행
    print("\n" + "-"*60)
    print("배치 발행")
    print("-"*60)
    client.publisher.publish_batch(messages)

    # 통계
    print("\n" + "="*60)
    print("GCP Pub/Sub 데모 완료")
    print("="*60)
    stats = client.get_statistics()
    print(f"발행된 메시지: {stats['published']}개")


def demo_command_control():
    """명령 및 제어 데모"""
    print("\n" + "="*60)
    print("명령 및 제어 데모")
    print("="*60)

    device_id = "smart-device-001"

    # 명령 생성 예시
    print("\n[클라우드 → 디바이스] 명령 전송")

    # 1. LED 제어 명령
    led_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="set_led",
        parameters={"color": "red", "brightness": 80}
    )
    print(f"\n1. LED 제어 명령:")
    print(json.dumps(led_command, indent=2))

    # 2. 설정 변경 명령
    config_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="update_config",
        parameters={"report_interval": 60, "threshold_temp": 30}
    )
    print(f"\n2. 설정 변경 명령:")
    print(json.dumps(config_command, indent=2))

    # 3. 펌웨어 업데이트 명령
    firmware_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="update_firmware",
        parameters={"version": "2.0.0", "url": "https://example.com/firmware.bin"}
    )
    print(f"\n3. 펌웨어 업데이트 명령:")
    print(json.dumps(firmware_command, indent=2))

    # 응답 예시
    print("\n[디바이스 → 클라우드] 명령 응답")

    response = MQTTMessageFormat.create_response(
        device_id=device_id,
        command_id=led_command["command_id"],
        status="success",
        result={"led_state": "on", "color": "red", "brightness": 80}
    )
    print(json.dumps(response, indent=2))


# ==============================================================================
# 메인 실행
# ==============================================================================

def main():
    """메인 함수"""
    print("클라우드 IoT 통합 - 시뮬레이션 모드")
    print("="*60)
    print("이 프로그램은 실제 클라우드 계정 없이 시뮬레이션으로 동작합니다.")
    print()

    # 메뉴
    print("데모 시나리오:")
    print("  1. AWS IoT Core")
    print("  2. GCP Pub/Sub")
    print("  3. 명령 및 제어")
    print("  4. 전체 실행")
    print()

    choice = input("선택 (1-4, 기본값=4): ").strip() or "4"

    if choice == "1":
        demo_aws_iot()
    elif choice == "2":
        demo_gcp_pubsub()
    elif choice == "3":
        demo_command_control()
    elif choice == "4":
        demo_aws_iot()
        time.sleep(2)
        demo_gcp_pubsub()
        time.sleep(2)
        demo_command_control()
    else:
        print("잘못된 선택")


if __name__ == "__main__":
    main()
