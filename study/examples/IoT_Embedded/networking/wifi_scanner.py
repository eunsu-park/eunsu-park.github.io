#!/usr/bin/env python3
"""
WiFi 네트워크 스캐너 및 소켓 프로그래밍 예제

이 스크립트는 다음 기능을 제공합니다:
1. WiFi 네트워크 스캔 (시뮬레이션 모드 지원)
2. TCP 클라이언트/서버 예제
3. HTTP 클라이언트로 센서 데이터 전송
4. 네트워크 연결 상태 모니터링
5. 네트워크 오류 처리 및 재시도 로직

참고: content/ko/IoT_Embedded/04_WiFi_Networking.md
"""

import subprocess
import re
import socket
import json
import time
import random
import sys
from typing import Optional, List, Dict
from datetime import datetime


# ============================================================================
# WiFi 네트워크 스캔
# ============================================================================

def is_raspberry_pi() -> bool:
    """라즈베리파이 환경인지 확인"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'BCM' in cpuinfo or 'Raspberry' in cpuinfo
    except FileNotFoundError:
        return False


def get_wifi_info() -> dict:
    """
    현재 WiFi 연결 정보 조회

    Returns:
        dict: WiFi 정보 (ssid, ip_addresses, signal_dbm, mac_address)
    """
    info = {}

    if not is_raspberry_pi():
        # 시뮬레이션 모드
        print("[시뮬레이션] 실제 라즈베리파이가 아닙니다. 가상 데이터 사용")
        info = {
            'ssid': 'SimulatedWiFi',
            'ip_addresses': ['192.168.1.100'],
            'signal_dbm': -45,
            'mac_address': 'AA:BB:CC:DD:EE:FF'
        }
        return info

    try:
        # SSID 조회
        result = subprocess.run(
            ['iwgetid', '-r'],
            capture_output=True,
            text=True,
            timeout=5
        )
        info['ssid'] = result.stdout.strip()

        # IP 주소 조회
        result = subprocess.run(
            ['hostname', '-I'],
            capture_output=True,
            text=True,
            timeout=5
        )
        ips = result.stdout.strip().split()
        info['ip_addresses'] = ips

        # 신호 강도 조회
        result = subprocess.run(
            ['iwconfig', 'wlan0'],
            capture_output=True,
            text=True,
            timeout=5
        )
        match = re.search(r'Signal level=(-?\d+)', result.stdout)
        if match:
            info['signal_dbm'] = int(match.group(1))

        # MAC 주소
        result = subprocess.run(
            ['cat', '/sys/class/net/wlan0/address'],
            capture_output=True,
            text=True,
            timeout=5
        )
        info['mac_address'] = result.stdout.strip()

    except Exception as e:
        info['error'] = str(e)

    return info


def get_wifi_networks() -> List[dict]:
    """
    주변 WiFi 네트워크 스캔

    Returns:
        list: 네트워크 정보 리스트 (ssid, quality, signal_dbm)
    """
    networks = []

    if not is_raspberry_pi():
        # 시뮬레이션 모드
        print("[시뮬레이션] WiFi 스캔 결과 생성 중...")
        networks = [
            {'ssid': 'HomeNetwork', 'quality': '70/70', 'signal_dbm': -35},
            {'ssid': 'OfficeWiFi', 'quality': '65/70', 'signal_dbm': -42},
            {'ssid': 'GuestNetwork', 'quality': '50/70', 'signal_dbm': -58},
            {'ssid': 'Neighbor_2.4G', 'quality': '40/70', 'signal_dbm': -65},
            {'ssid': 'CafeWiFi', 'quality': '30/70', 'signal_dbm': -75},
        ]
        return networks

    try:
        result = subprocess.run(
            ['sudo', 'iwlist', 'wlan0', 'scan'],
            capture_output=True,
            text=True,
            timeout=10
        )

        current_network = {}
        for line in result.stdout.split('\n'):
            if 'ESSID:' in line:
                ssid = re.search(r'ESSID:"(.+)"', line)
                if ssid and current_network:
                    networks.append(current_network)
                current_network = {'ssid': ssid.group(1) if ssid else ''}

            elif 'Quality=' in line:
                quality = re.search(r'Quality=(\d+)/(\d+)', line)
                if quality:
                    current_network['quality'] = f"{quality.group(1)}/{quality.group(2)}"

                signal = re.search(r'Signal level=(-?\d+)', line)
                if signal:
                    current_network['signal_dbm'] = int(signal.group(1))

        if current_network:
            networks.append(current_network)

    except Exception as e:
        print(f"스캔 실패: {e}")
        print("시뮬레이션 모드로 전환합니다...")
        return get_wifi_networks()  # 재귀 호출로 시뮬레이션 모드 사용

    return networks


def monitor_wifi_signal(threshold_dbm: int = -70, interval: int = 5, duration: int = 30):
    """
    WiFi 신호 강도 모니터링

    Args:
        threshold_dbm: 경고 임계값 (dBm)
        interval: 확인 간격 (초)
        duration: 모니터링 기간 (초)
    """
    print(f"=== WiFi 신호 모니터링 시작 ===")
    print(f"경고 임계값: {threshold_dbm} dBm")
    print(f"확인 간격: {interval}초, 총 {duration}초")
    print()

    start_time = time.time()

    while time.time() - start_time < duration:
        info = get_wifi_info()

        if 'error' in info:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 오류: {info['error']}")
        else:
            signal = info.get('signal_dbm', 0)
            ssid = info.get('ssid', 'Unknown')

            status = "양호" if signal > threshold_dbm else "⚠️  경고"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ssid}: {signal} dBm ({status})")

            if signal <= threshold_dbm:
                print(f"  ⚠️  신호가 약합니다! ({signal} dBm <= {threshold_dbm} dBm)")

        time.sleep(interval)

    print("\n모니터링 종료")


# ============================================================================
# TCP 소켓 프로그래밍
# ============================================================================

def tcp_server(host: str = '0.0.0.0', port: int = 9999):
    """
    TCP 서버 - 센서 데이터 수신

    Args:
        host: 바인드할 IP 주소
        port: 포트 번호
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # 주소 재사용 허용
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server.bind((host, port))
        server.listen(5)

        print(f"TCP 서버 시작: {host}:{port}")
        print("클라이언트 연결 대기 중... (Ctrl+C로 종료)")

        try:
            while True:
                client, address = server.accept()
                print(f"\n클라이언트 연결: {address}")

                with client:
                    while True:
                        data = client.recv(1024)
                        if not data:
                            break

                        try:
                            # JSON 데이터 파싱
                            message = json.loads(data.decode('utf-8'))
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] 수신: {message}")

                            # 응답 전송
                            response = {
                                "status": "ok",
                                "received": message.get("sensor_id"),
                                "timestamp": time.time()
                            }
                            client.sendall(json.dumps(response).encode('utf-8'))

                        except json.JSONDecodeError:
                            print(f"잘못된 JSON: {data}")

                print(f"클라이언트 연결 종료: {address}")

        except KeyboardInterrupt:
            print("\n서버 종료")


def tcp_client(server_host: str, server_port: int = 9999, count: int = 10):
    """
    TCP 클라이언트 - 센서 데이터 전송

    Args:
        server_host: 서버 IP 주소
        server_port: 서버 포트
        count: 전송 횟수
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            print(f"서버 연결 중: {server_host}:{server_port}")
            client.connect((server_host, server_port))
            print(f"서버 연결 성공!")

            sensor_id = "temp_sensor_01"

            for i in range(count):
                # 센서 데이터 생성 (시뮬레이션)
                data = {
                    "sensor_id": sensor_id,
                    "temperature": round(random.uniform(20, 30), 1),
                    "humidity": round(random.uniform(40, 70), 1),
                    "timestamp": time.time(),
                    "sequence": i + 1
                }

                # 전송
                message = json.dumps(data).encode('utf-8')
                client.sendall(message)
                print(f"[{i+1}/{count}] 전송: 온도={data['temperature']}°C, 습도={data['humidity']}%")

                # 응답 수신
                try:
                    client.settimeout(5.0)
                    response = client.recv(1024)
                    if response:
                        resp_data = json.loads(response.decode('utf-8'))
                        print(f"         응답: {resp_data['status']}")
                except socket.timeout:
                    print(f"         응답 타임아웃")

                if i < count - 1:
                    time.sleep(2)

            print("\n전송 완료")

    except ConnectionRefusedError:
        print(f"오류: 서버에 연결할 수 없습니다 ({server_host}:{server_port})")
        print("서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"오류: {e}")


# ============================================================================
# HTTP 클라이언트
# ============================================================================

def send_sensor_data_http(api_base: str, sensor_id: str, data: dict) -> bool:
    """
    HTTP POST로 센서 데이터 전송

    Args:
        api_base: API 베이스 URL
        sensor_id: 센서 ID
        data: 전송할 데이터

    Returns:
        bool: 성공 여부
    """
    try:
        import requests
    except ImportError:
        print("requests 라이브러리가 필요합니다: pip install requests")
        return False

    url = f"{api_base}/sensors/{sensor_id}/data"

    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )

        if response.status_code in (200, 201):
            print(f"데이터 전송 성공: {data}")
            return True
        else:
            print(f"전송 실패: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"네트워크 오류: {e}")
        return False


def http_client_with_retry(api_base: str, sensor_id: str, retries: int = 3):
    """
    재시도 로직이 있는 HTTP 클라이언트

    Args:
        api_base: API 베이스 URL
        sensor_id: 센서 ID
        retries: 재시도 횟수
    """
    print(f"=== HTTP 클라이언트 (재시도: {retries}회) ===")
    print(f"API: {api_base}")
    print(f"센서 ID: {sensor_id}\n")

    for i in range(10):
        data = {
            "temperature": round(random.uniform(20, 30), 1),
            "humidity": round(random.uniform(40, 70), 1),
            "timestamp": int(time.time())
        }

        success = False
        for attempt in range(retries):
            if send_sensor_data_http(api_base, sensor_id, data):
                success = True
                break
            else:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  재시도 대기: {wait_time}초...")
                    time.sleep(wait_time)

        if not success:
            print(f"  ❌ {retries}번 재시도 후 실패")

        time.sleep(5)


# ============================================================================
# 네트워크 연결 상태 확인
# ============================================================================

def check_internet_connection(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """
    인터넷 연결 확인

    Args:
        host: 테스트할 호스트 (기본: Google DNS)
        port: 포트
        timeout: 타임아웃 (초)

    Returns:
        bool: 연결 가능 여부
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def get_local_ip() -> str:
    """
    로컬 IP 주소 조회

    Returns:
        str: IP 주소
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def network_status():
    """네트워크 상태 확인"""
    print("=== 네트워크 상태 ===\n")

    # 로컬 IP
    local_ip = get_local_ip()
    print(f"로컬 IP: {local_ip}")

    # 인터넷 연결
    internet = check_internet_connection()
    print(f"인터넷 연결: {'✓ 연결됨' if internet else '✗ 연결 안됨'}")

    # WiFi 정보
    wifi_info = get_wifi_info()
    if 'error' not in wifi_info:
        print(f"\nWiFi SSID: {wifi_info.get('ssid', 'N/A')}")
        print(f"신호 강도: {wifi_info.get('signal_dbm', 'N/A')} dBm")
        print(f"MAC 주소: {wifi_info.get('mac_address', 'N/A')}")


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("WiFi 네트워크 스캐너 및 소켓 프로그래밍 예제")
        print("\n사용법:")
        print("  python wifi_scanner.py info         - WiFi 정보 조회")
        print("  python wifi_scanner.py scan         - WiFi 네트워크 스캔")
        print("  python wifi_scanner.py monitor      - WiFi 신호 모니터링")
        print("  python wifi_scanner.py status       - 네트워크 상태 확인")
        print("  python wifi_scanner.py server       - TCP 서버 시작")
        print("  python wifi_scanner.py client <IP>  - TCP 클라이언트 시작")
        print("\n예제:")
        print("  python wifi_scanner.py info")
        print("  python wifi_scanner.py client 192.168.1.100")
        return

    command = sys.argv[1].lower()

    if command == 'info':
        print("=== WiFi 연결 정보 ===")
        info = get_wifi_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

    elif command == 'scan':
        print("=== 주변 WiFi 네트워크 ===")
        networks = get_wifi_networks()
        print(f"발견된 네트워크: {len(networks)}개\n")
        for net in networks:
            ssid = net.get('ssid', 'Unknown')
            signal = net.get('signal_dbm', 'N/A')
            quality = net.get('quality', 'N/A')
            print(f"  {ssid:20} - 신호: {signal:4} dBm, 품질: {quality}")

    elif command == 'monitor':
        monitor_wifi_signal(threshold_dbm=-70, interval=5, duration=30)

    elif command == 'status':
        network_status()

    elif command == 'server':
        tcp_server()

    elif command == 'client':
        if len(sys.argv) < 3:
            print("오류: 서버 IP 주소를 입력하세요")
            print("예제: python wifi_scanner.py client 192.168.1.100")
            return
        server_ip = sys.argv[2]
        tcp_client(server_ip, count=10)

    else:
        print(f"알 수 없는 명령: {command}")
        print("'python wifi_scanner.py'를 실행하여 도움말을 확인하세요")


if __name__ == "__main__":
    main()
