#!/usr/bin/env python3
"""
BLE (Bluetooth Low Energy) 장치 스캐너 및 GATT 클라이언트 예제

이 스크립트는 다음 기능을 제공합니다:
1. BLE 장치 스캔 (시뮬레이션 모드 지원)
2. GATT 서비스 및 특성 탐색
3. BLE 특성 값 읽기
4. BLE 알림(Notification) 수신
5. 센서 데이터 수신 예제

참고: content/ko/IoT_Embedded/05_BLE_Connectivity.md

주의: 실제 BLE 기능을 사용하려면 bleak 라이브러리가 필요합니다.
      pip install bleak
"""

import asyncio
import sys
import time
import random
import struct
from typing import Optional, List, Dict, Callable
from datetime import datetime


# ============================================================================
# BLE 라이브러리 임포트 (선택적)
# ============================================================================

try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("⚠️  bleak 라이브러리를 찾을 수 없습니다.")
    print("   시뮬레이션 모드로 실행됩니다.")
    print("   실제 BLE 기능을 사용하려면 'pip install bleak'를 실행하세요.\n")


# ============================================================================
# 표준 BLE UUID
# ============================================================================

class BLE_UUID:
    """표준 BLE 서비스 및 특성 UUID"""

    # 표준 서비스 UUID (16-bit)
    GENERIC_ACCESS = "00001800-0000-1000-8000-00805f9b34fb"
    GENERIC_ATTRIBUTE = "00001801-0000-1000-8000-00805f9b34fb"
    DEVICE_INFORMATION = "0000180a-0000-1000-8000-00805f9b34fb"
    BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb"
    ENVIRONMENTAL_SENSING = "0000181a-0000-1000-8000-00805f9b34fb"
    HEART_RATE = "0000180d-0000-1000-8000-00805f9b34fb"

    # 표준 특성 UUID
    DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"
    BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
    TEMPERATURE = "00002a6e-0000-1000-8000-00805f9b34fb"
    HUMIDITY = "00002a6f-0000-1000-8000-00805f9b34fb"
    HEART_RATE_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"

    @staticmethod
    def uuid_16_to_128(uuid_16: str) -> str:
        """
        16-bit UUID를 128-bit BLE 기본 UUID로 변환

        Args:
            uuid_16: 16-bit UUID (예: "0x180F")

        Returns:
            str: 128-bit UUID
        """
        base_uuid = "00000000-0000-1000-8000-00805f9b34fb"
        uuid_16_clean = uuid_16.replace("0x", "").lower()
        return f"0000{uuid_16_clean}{base_uuid[8:]}"


# ============================================================================
# 시뮬레이션 모드
# ============================================================================

class SimulatedBLEDevice:
    """시뮬레이션용 BLE 장치"""

    def __init__(self, name: str, address: str, rssi: int):
        self.name = name
        self.address = address
        self.rssi = rssi

    def __repr__(self):
        return f"SimulatedBLEDevice(name='{self.name}', address='{self.address}', rssi={self.rssi})"


def simulate_ble_scan(timeout: float = 10.0) -> List[SimulatedBLEDevice]:
    """
    BLE 스캔 시뮬레이션

    Args:
        timeout: 스캔 시간 (초)

    Returns:
        list: 시뮬레이션된 BLE 장치 리스트
    """
    print(f"[시뮬레이션] BLE 장치 스캔 중... ({timeout}초)")
    time.sleep(2)  # 스캔 시뮬레이션

    devices = [
        SimulatedBLEDevice("TempSensor-01", "AA:BB:CC:DD:EE:01", -45),
        SimulatedBLEDevice("HeartRate-BLE", "AA:BB:CC:DD:EE:02", -52),
        SimulatedBLEDevice("Battery-Monitor", "AA:BB:CC:DD:EE:03", -38),
        SimulatedBLEDevice("EnvSensor", "AA:BB:CC:DD:EE:04", -61),
        SimulatedBLEDevice("Smart-Watch", "AA:BB:CC:DD:EE:05", -55),
        SimulatedBLEDevice(None, "AA:BB:CC:DD:EE:06", -72),  # 이름 없는 장치
    ]

    return devices


def simulate_read_characteristic(char_uuid: str) -> bytes:
    """
    BLE 특성 읽기 시뮬레이션

    Args:
        char_uuid: 특성 UUID

    Returns:
        bytes: 시뮬레이션 데이터
    """
    if BLE_UUID.BATTERY_LEVEL in char_uuid:
        # 배터리 레벨 (0-100%)
        return bytes([random.randint(50, 100)])

    elif BLE_UUID.TEMPERATURE in char_uuid:
        # 온도 (0.01도 단위, 16-bit 정수)
        temp = random.uniform(20.0, 30.0)
        temp_raw = int(temp * 100)
        return struct.pack('<h', temp_raw)

    elif BLE_UUID.HUMIDITY in char_uuid:
        # 습도 (0.01% 단위, 16-bit 정수)
        humidity = random.uniform(40.0, 70.0)
        humidity_raw = int(humidity * 100)
        return struct.pack('<H', humidity_raw)

    else:
        # 기본값
        return b'\x00\x00'


# ============================================================================
# BLE 스캔 함수
# ============================================================================

async def scan_ble_devices(timeout: float = 10.0, use_simulation: bool = False) -> List:
    """
    BLE 장치 스캔

    Args:
        timeout: 스캔 시간 (초)
        use_simulation: 강제로 시뮬레이션 모드 사용

    Returns:
        list: 발견된 BLE 장치 리스트
    """
    if not BLEAK_AVAILABLE or use_simulation:
        return simulate_ble_scan(timeout)

    print(f"BLE 장치 스캔 중... ({timeout}초)")

    try:
        devices = await BleakScanner.discover(timeout=timeout)
        return devices
    except Exception as e:
        print(f"스캔 오류: {e}")
        print("시뮬레이션 모드로 전환합니다...")
        return simulate_ble_scan(timeout)


async def scan_with_filter(name_filter: Optional[str] = None, timeout: float = 10.0) -> List:
    """
    필터링된 BLE 스캔

    Args:
        name_filter: 장치 이름 필터 (부분 일치)
        timeout: 스캔 시간

    Returns:
        list: 필터링된 장치 리스트
    """
    devices = await scan_ble_devices(timeout)

    if name_filter:
        devices = [d for d in devices if d.name and name_filter.lower() in d.name.lower()]

    return devices


async def continuous_scan(duration: float = 30.0, callback: Optional[Callable] = None):
    """
    연속 BLE 스캔

    Args:
        duration: 스캔 기간 (초)
        callback: 장치 발견 시 호출할 콜백 함수
    """
    if not BLEAK_AVAILABLE:
        print("[시뮬레이션] 연속 스캔은 시뮬레이션 모드에서 지원되지 않습니다.")
        devices = simulate_ble_scan(duration)
        for device in devices:
            print(f"발견: {device.name} ({device.address}) - RSSI: {device.rssi} dBm")
        return

    def detection_callback(device, advertisement_data):
        print(f"발견: {device.name or 'Unknown'} ({device.address}) - RSSI: {device.rssi} dBm")
        if callback:
            callback(device, advertisement_data)

    scanner = BleakScanner(detection_callback=detection_callback)

    print(f"연속 스캔 시작 ({duration}초)")
    await scanner.start()
    await asyncio.sleep(duration)
    await scanner.stop()
    print("스캔 종료")


# ============================================================================
# BLE 연결 및 탐색
# ============================================================================

async def connect_and_explore(address: str, use_simulation: bool = False):
    """
    BLE 장치 연결 및 서비스/특성 탐색

    Args:
        address: BLE 장치 MAC 주소
        use_simulation: 시뮬레이션 모드 사용
    """
    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[시뮬레이션] 연결 중: {address}")
        print(f"[시뮬레이션] 연결됨!")
        print(f"\n서비스: {BLE_UUID.ENVIRONMENTAL_SENSING}")
        print(f"  설명: Environmental Sensing")
        print(f"    특성: {BLE_UUID.TEMPERATURE}")
        print(f"      속성: ['read', 'notify']")
        print(f"      값: {simulate_read_characteristic(BLE_UUID.TEMPERATURE).hex()}")
        print(f"    특성: {BLE_UUID.HUMIDITY}")
        print(f"      속성: ['read', 'notify']")
        print(f"      값: {simulate_read_characteristic(BLE_UUID.HUMIDITY).hex()}")
        return

    print(f"연결 중: {address}")

    try:
        async with BleakClient(address) as client:
            print(f"연결됨! MTU: {client.mtu_size}")

            # 서비스 탐색
            for service in client.services:
                print(f"\n서비스: {service.uuid}")
                print(f"  설명: {service.description}")

                # 특성 탐색
                for char in service.characteristics:
                    print(f"    특성: {char.uuid}")
                    print(f"      속성: {char.properties}")

                    # 읽기 가능하면 값 읽기
                    if "read" in char.properties:
                        try:
                            value = await client.read_gatt_char(char.uuid)
                            print(f"      값: {value.hex()}")
                        except Exception as e:
                            print(f"      읽기 실패: {e}")

    except Exception as e:
        print(f"연결 오류: {e}")
        print("시뮬레이션 모드로 재시도...")
        await connect_and_explore(address, use_simulation=True)


# ============================================================================
# BLE 센서 데이터 읽기
# ============================================================================

async def read_sensor_data(address: str, use_simulation: bool = False) -> Dict:
    """
    BLE 센서 데이터 읽기

    Args:
        address: BLE 장치 주소
        use_simulation: 시뮬레이션 모드

    Returns:
        dict: 센서 데이터
    """
    result = {}

    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[시뮬레이션] 센서 데이터 읽기: {address}")

        # 배터리
        battery_data = simulate_read_characteristic(BLE_UUID.BATTERY_LEVEL)
        result['battery'] = battery_data[0]

        # 온도
        temp_data = simulate_read_characteristic(BLE_UUID.TEMPERATURE)
        temp_raw = struct.unpack('<h', temp_data)[0]
        result['temperature'] = temp_raw * 0.01

        # 습도
        humidity_data = simulate_read_characteristic(BLE_UUID.HUMIDITY)
        humidity_raw = struct.unpack('<H', humidity_data)[0]
        result['humidity'] = humidity_raw * 0.01

        return result

    try:
        async with BleakClient(address) as client:
            # 배터리 레벨
            try:
                data = await client.read_gatt_char(BLE_UUID.BATTERY_LEVEL)
                result['battery'] = data[0]
            except Exception:
                pass

            # 온도
            try:
                data = await client.read_gatt_char(BLE_UUID.TEMPERATURE)
                temp_raw = struct.unpack('<h', data[:2])[0]
                result['temperature'] = temp_raw * 0.01
            except Exception:
                pass

            # 습도
            try:
                data = await client.read_gatt_char(BLE_UUID.HUMIDITY)
                humidity_raw = struct.unpack('<H', data[:2])[0]
                result['humidity'] = humidity_raw * 0.01
            except Exception:
                pass

    except Exception as e:
        print(f"읽기 오류: {e}")
        print("시뮬레이션 모드로 재시도...")
        return await read_sensor_data(address, use_simulation=True)

    return result


# ============================================================================
# BLE 알림 수신
# ============================================================================

def create_notification_handler(sensor_type: str):
    """
    알림 핸들러 생성

    Args:
        sensor_type: 센서 타입 ('temperature', 'humidity', etc.)

    Returns:
        function: 알림 핸들러 함수
    """
    def handler(sender, data):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 수신 ({sensor_type}): {data.hex()}")

        if sensor_type == 'temperature':
            temp_raw = struct.unpack('<h', data[:2])[0]
            temp = temp_raw * 0.01
            print(f"  온도: {temp:.2f}°C")

        elif sensor_type == 'humidity':
            humidity_raw = struct.unpack('<H', data[:2])[0]
            humidity = humidity_raw * 0.01
            print(f"  습도: {humidity:.2f}%")

        elif sensor_type == 'battery':
            battery = data[0]
            print(f"  배터리: {battery}%")

        elif sensor_type == 'heart_rate':
            flags = data[0]
            if flags & 0x01:  # 16-bit heart rate
                hr = int.from_bytes(data[1:3], 'little')
            else:  # 8-bit heart rate
                hr = data[1]
            print(f"  심박수: {hr} bpm")

    return handler


async def subscribe_notifications(
    address: str,
    char_uuid: str,
    sensor_type: str = 'unknown',
    duration: float = 60,
    use_simulation: bool = False
):
    """
    BLE 알림 구독

    Args:
        address: BLE 장치 주소
        char_uuid: 특성 UUID
        sensor_type: 센서 타입
        duration: 구독 기간 (초)
        use_simulation: 시뮬레이션 모드
    """
    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[시뮬레이션] 알림 구독: {address}")
        print(f"[시뮬레이션] 특성: {char_uuid}")
        print(f"\n{duration}초 동안 시뮬레이션 데이터 수신...\n")

        handler = create_notification_handler(sensor_type)

        for i in range(int(duration / 2)):
            # 시뮬레이션 데이터 생성
            data = simulate_read_characteristic(char_uuid)
            handler(char_uuid, data)
            await asyncio.sleep(2)

        print("\n구독 종료")
        return

    print(f"연결 중: {address}")

    try:
        async with BleakClient(address) as client:
            print(f"연결됨!")

            handler = create_notification_handler(sensor_type)

            # 알림 시작
            await client.start_notify(char_uuid, handler)
            print(f"알림 구독 시작: {char_uuid}")
            print(f"{duration}초 동안 수신 중...\n")

            # 지정된 시간 동안 수신
            await asyncio.sleep(duration)

            # 알림 중지
            await client.stop_notify(char_uuid)
            print("\n알림 구독 종료")

    except Exception as e:
        print(f"구독 오류: {e}")
        print("시뮬레이션 모드로 재시도...")
        await subscribe_notifications(address, char_uuid, sensor_type, duration, use_simulation=True)


# ============================================================================
# BLE 센서 모니터 클래스
# ============================================================================

class BLESensorMonitor:
    """BLE 환경 센서 모니터링 클래스"""

    def __init__(self, device_address: Optional[str] = None, use_simulation: bool = False):
        self.device_address = device_address
        self.use_simulation = use_simulation or not BLEAK_AVAILABLE
        self.data_buffer = []

    async def start_monitoring(self, duration: float = 60):
        """
        모니터링 시작

        Args:
            duration: 모니터링 기간 (초)
        """
        if not self.device_address:
            print("오류: 장치 주소가 지정되지 않았습니다.")
            return

        print(f"=== BLE 센서 모니터링 시작 ===")
        print(f"장치: {self.device_address}")
        print(f"기간: {duration}초")
        print(f"모드: {'시뮬레이션' if self.use_simulation else '실제'}\n")

        if self.use_simulation:
            # 시뮬레이션 모니터링
            for i in range(int(duration / 5)):
                data = await read_sensor_data(self.device_address, use_simulation=True)
                timestamp = datetime.now()

                print(f"[{timestamp.strftime('%H:%M:%S')}] 수신:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
                    self.data_buffer.append({
                        'type': key,
                        'value': value,
                        'timestamp': timestamp
                    })

                await asyncio.sleep(5)
        else:
            # 실제 BLE 모니터링
            await subscribe_notifications(
                self.device_address,
                BLE_UUID.TEMPERATURE,
                'temperature',
                duration / 2
            )

        print("\n=== 모니터링 종료 ===")
        print(f"수집된 데이터: {len(self.data_buffer)}개")

    def get_summary(self) -> Dict:
        """수집된 데이터 요약"""
        if not self.data_buffer:
            return {}

        summary = {}
        data_types = set(d['type'] for d in self.data_buffer)

        for dtype in data_types:
            values = [d['value'] for d in self.data_buffer if d['type'] == dtype]
            summary[dtype] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }

        return summary


# ============================================================================
# 메인 함수
# ============================================================================

def print_devices(devices: List):
    """장치 리스트 출력"""
    print(f"\n발견된 장치: {len(devices)}개\n")

    for i, device in enumerate(devices, 1):
        name = device.name or 'Unknown'
        address = device.address
        rssi = getattr(device, 'rssi', 'N/A')
        print(f"{i:2}. {name:20} - {address} (RSSI: {rssi} dBm)")


async def main_async():
    """비동기 메인 함수"""
    if len(sys.argv) < 2:
        print("BLE 장치 스캐너 및 GATT 클라이언트 예제")
        print("\n사용법:")
        print("  python ble_scanner.py scan              - BLE 장치 스캔")
        print("  python ble_scanner.py scan <필터>       - 이름 필터로 스캔")
        print("  python ble_scanner.py explore <주소>    - 장치 탐색")
        print("  python ble_scanner.py read <주소>       - 센서 데이터 읽기")
        print("  python ble_scanner.py notify <주소>     - 알림 수신")
        print("  python ble_scanner.py monitor <주소>    - 센서 모니터링")
        print("\n예제:")
        print("  python ble_scanner.py scan")
        print("  python ble_scanner.py scan temp")
        print("  python ble_scanner.py explore AA:BB:CC:DD:EE:FF")
        print("\n주의: bleak 라이브러리가 없으면 시뮬레이션 모드로 실행됩니다.")
        return

    command = sys.argv[1].lower()

    if command == 'scan':
        name_filter = sys.argv[2] if len(sys.argv) > 2 else None
        if name_filter:
            devices = await scan_with_filter(name_filter, timeout=10.0)
        else:
            devices = await scan_ble_devices(timeout=10.0)
        print_devices(devices)

    elif command == 'explore':
        if len(sys.argv) < 3:
            print("오류: 장치 주소를 입력하세요")
            print("예제: python ble_scanner.py explore AA:BB:CC:DD:EE:FF")
            return
        address = sys.argv[2]
        await connect_and_explore(address)

    elif command == 'read':
        if len(sys.argv) < 3:
            print("오류: 장치 주소를 입력하세요")
            return
        address = sys.argv[2]
        data = await read_sensor_data(address)
        print("\n=== 센서 데이터 ===")
        for key, value in data.items():
            print(f"  {key}: {value}")

    elif command == 'notify':
        if len(sys.argv) < 3:
            print("오류: 장치 주소를 입력하세요")
            return
        address = sys.argv[2]
        await subscribe_notifications(
            address,
            BLE_UUID.TEMPERATURE,
            'temperature',
            duration=30
        )

    elif command == 'monitor':
        if len(sys.argv) < 3:
            print("오류: 장치 주소를 입력하세요")
            return
        address = sys.argv[2]
        monitor = BLESensorMonitor(address)
        await monitor.start_monitoring(duration=30)

        # 요약 출력
        summary = monitor.get_summary()
        if summary:
            print("\n=== 데이터 요약 ===")
            for sensor, stats in summary.items():
                print(f"{sensor}:")
                print(f"  최소: {stats['min']:.2f}")
                print(f"  최대: {stats['max']:.2f}")
                print(f"  평균: {stats['avg']:.2f}")
                print(f"  개수: {stats['count']}")

    else:
        print(f"알 수 없는 명령: {command}")
        print("'python ble_scanner.py'를 실행하여 도움말을 확인하세요")


def main():
    """메인 함수"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n사용자 중단")


if __name__ == "__main__":
    main()
