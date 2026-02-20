#!/usr/bin/env python3
"""
IoT Flask REST API 서버
간단한 센서 데이터 수집 및 장치 제어 API

참고: content/ko/IoT_Embedded/07_HTTP_REST_for_IoT.md
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import uuid
import sqlite3
import os

app = Flask(__name__)
CORS(app)  # CORS 활성화 (웹 클라이언트 지원)

# === 데이터베이스 설정 ===
DB_PATH = "iot_data.db"
USE_SQLITE = True  # False로 설정하면 메모리 저장소 사용


def init_db():
    """SQLite 데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 센서 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            location TEXT,
            status TEXT DEFAULT 'active',
            registered_at TEXT
        )
    """)

    # 센서 데이터 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id TEXT PRIMARY KEY,
            sensor_id TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        )
    """)

    # 장치 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            status TEXT DEFAULT 'offline',
            created_at TEXT,
            last_seen TEXT
        )
    """)

    conn.commit()
    conn.close()


# 메모리 저장소 (시뮬레이션 모드)
memory_store = {
    'sensors': {},
    'sensor_readings': [],
    'devices': {}
}


# === 헬퍼 함수 ===
def get_db_connection():
    """데이터베이스 연결 획득"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def dict_from_row(row):
    """sqlite3.Row를 딕셔너리로 변환"""
    return dict(zip(row.keys(), row))


# === API 엔드포인트 ===

@app.route('/')
def index():
    """API 정보"""
    return jsonify({
        "name": "IoT REST API Server",
        "version": "1.0",
        "storage": "SQLite" if USE_SQLITE else "Memory",
        "endpoints": {
            "/": "GET - API 정보",
            "/health": "GET - 헬스 체크",
            "/api/sensors": "GET, POST - 센서 목록/등록",
            "/api/sensors/<id>": "GET - 센서 정보 조회",
            "/api/sensors/<id>/data": "GET, POST - 센서 데이터 조회/등록",
            "/api/sensors/<id>/latest": "GET - 최신 센서 데이터",
            "/api/sensors/<id>/stats": "GET - 센서 통계",
            "/api/devices": "GET, POST - 장치 목록/등록",
            "/api/devices/<id>": "GET, PUT, DELETE - 장치 조회/수정/삭제",
            "/api/devices/<id>/command": "POST - 장치 명령 전송"
        }
    })


@app.route('/health')
def health():
    """헬스 체크"""
    if USE_SQLITE:
        # DB 연결 확인
        try:
            conn = get_db_connection()
            conn.close()
            status = "healthy"
        except Exception as e:
            status = f"unhealthy: {str(e)}"
    else:
        status = "healthy (memory mode)"

    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "storage": "SQLite" if USE_SQLITE else "Memory"
    })


# === 센서 API ===

@app.route('/api/sensors', methods=['GET'])
def list_sensors():
    """등록된 센서 목록 조회"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensors")
        sensors = [dict_from_row(row) for row in cursor.fetchall()]
        conn.close()
    else:
        sensors = list(memory_store['sensors'].values())

    return jsonify({
        "sensors": sensors,
        "count": len(sensors)
    })


@app.route('/api/sensors', methods=['POST'])
def register_sensor():
    """새 센서 등록"""
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({"error": "name is required"}), 400

    sensor_id = str(uuid.uuid4())[:8]
    sensor = {
        "id": sensor_id,
        "name": data['name'],
        "type": data.get('type', 'generic'),
        "location": data.get('location', 'unknown'),
        "status": "active",
        "registered_at": datetime.now().isoformat()
    }

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sensors (id, name, type, location, status, registered_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sensor['id'], sensor['name'], sensor['type'],
              sensor['location'], sensor['status'], sensor['registered_at']))
        conn.commit()
        conn.close()
    else:
        memory_store['sensors'][sensor_id] = sensor

    return jsonify(sensor), 201


@app.route('/api/sensors/<sensor_id>', methods=['GET'])
def get_sensor(sensor_id):
    """센서 정보 조회"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensors WHERE id = ?", (sensor_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Sensor not found"}), 404

        sensor = dict_from_row(row)
    else:
        sensor = memory_store['sensors'].get(sensor_id)
        if not sensor:
            return jsonify({"error": "Sensor not found"}), 404

    return jsonify(sensor)


@app.route('/api/sensors/<sensor_id>/data', methods=['POST'])
def post_sensor_data(sensor_id):
    """센서 데이터 수신"""
    import json

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    reading_id = str(uuid.uuid4())
    reading = {
        "id": reading_id,
        "sensor_id": sensor_id,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

    if USE_SQLITE:
        # 센서 자동 등록 (존재하지 않는 경우)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sensors WHERE id = ?", (sensor_id,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO sensors (id, name, type, status, registered_at)
                VALUES (?, ?, ?, ?, ?)
            """, (sensor_id, f"auto_{sensor_id}", "generic",
                  "active", datetime.now().isoformat()))

        # 센서 데이터 저장
        cursor.execute("""
            INSERT INTO sensor_readings (id, sensor_id, data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (reading['id'], reading['sensor_id'],
              json.dumps(reading['data']), reading['timestamp']))
        conn.commit()
        conn.close()
    else:
        # 센서 자동 등록
        if sensor_id not in memory_store['sensors']:
            memory_store['sensors'][sensor_id] = {
                "id": sensor_id,
                "name": f"auto_{sensor_id}",
                "type": "generic",
                "status": "active",
                "registered_at": datetime.now().isoformat()
            }

        memory_store['sensor_readings'].append(reading)

        # 최근 1000개만 유지
        if len(memory_store['sensor_readings']) > 1000:
            memory_store['sensor_readings'].pop(0)

    return jsonify({"status": "ok", "reading_id": reading['id']}), 201


@app.route('/api/sensors/<sensor_id>/data', methods=['GET'])
def get_sensor_data(sensor_id):
    """센서 데이터 조회"""
    import json

    # 쿼리 파라미터
    limit = request.args.get('limit', 100, type=int)
    since = request.args.get('since', None)  # ISO timestamp

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM sensor_readings WHERE sensor_id = ?"
        params = [sensor_id]

        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        readings = []
        for row in rows:
            reading_dict = dict_from_row(row)
            reading_dict['data'] = json.loads(reading_dict['data'])
            readings.append(reading_dict)
    else:
        # 필터링
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

        if since:
            readings = [r for r in readings if r['timestamp'] > since]

        # 최신순 정렬 및 제한
        readings = sorted(readings, key=lambda x: x['timestamp'],
                         reverse=True)[:limit]

    return jsonify({
        "sensor_id": sensor_id,
        "readings": readings,
        "count": len(readings)
    })


@app.route('/api/sensors/<sensor_id>/latest', methods=['GET'])
def get_latest_reading(sensor_id):
    """최신 센서 데이터 조회"""
    import json

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sensor_readings
            WHERE sensor_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (sensor_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No data found"}), 404

        latest = dict_from_row(row)
        latest['data'] = json.loads(latest['data'])
    else:
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

        if not readings:
            return jsonify({"error": "No data found"}), 404

        latest = max(readings, key=lambda x: x['timestamp'])

    return jsonify(latest)


@app.route('/api/sensors/<sensor_id>/stats', methods=['GET'])
def get_sensor_stats(sensor_id):
    """센서 데이터 통계 (숫자 필드)"""
    import json

    field = request.args.get('field', 'temperature')

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data FROM sensor_readings WHERE sensor_id = ?
        """, (sensor_id,))
        rows = cursor.fetchall()
        conn.close()

        values = []
        for row in rows:
            data = json.loads(row['data'])
            if field in data:
                try:
                    values.append(float(data[field]))
                except (ValueError, TypeError):
                    pass
    else:
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

        values = []
        for r in readings:
            if field in r.get('data', {}):
                try:
                    values.append(float(r['data'][field]))
                except (ValueError, TypeError):
                    pass

    if not values:
        return jsonify({"error": f"No numeric data for field: {field}"}), 404

    stats = {
        "sensor_id": sensor_id,
        "field": field,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "latest": values[-1] if values else None
    }

    return jsonify(stats)


# === 장치 API ===

@app.route('/api/devices', methods=['GET'])
def list_devices():
    """장치 목록 조회"""
    # 필터링
    device_type = request.args.get('type')
    status = request.args.get('status')

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM devices WHERE 1=1"
        params = []

        if device_type:
            query += " AND type = ?"
            params.append(device_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        devices = [dict_from_row(row) for row in cursor.fetchall()]
        conn.close()
    else:
        devices = list(memory_store['devices'].values())

        if device_type:
            devices = [d for d in devices if d.get('type') == device_type]
        if status:
            devices = [d for d in devices if d.get('status') == status]

    return jsonify({
        "devices": devices,
        "total": len(devices)
    })


@app.route('/api/devices', methods=['POST'])
def create_device():
    """장치 등록"""
    data = request.get_json()

    required_fields = ['id', 'name', 'type']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    device_id = data['id']

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device already exists"}), 409
    else:
        if device_id in memory_store['devices']:
            return jsonify({"error": "Device already exists"}), 409

    device = {
        "id": device_id,
        "name": data['name'],
        "type": data['type'],
        "status": "offline",
        "created_at": datetime.now().isoformat(),
        "last_seen": None
    }

    if USE_SQLITE:
        cursor.execute("""
            INSERT INTO devices (id, name, type, status, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device['id'], device['name'], device['type'],
              device['status'], device['created_at'], device['last_seen']))
        conn.commit()
        conn.close()
    else:
        memory_store['devices'][device_id] = device

    return jsonify(device), 201


@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    """장치 정보 조회"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Device not found"}), 404

        device = dict_from_row(row)
    else:
        device = memory_store['devices'].get(device_id)
        if not device:
            return jsonify({"error": "Device not found"}), 404

    return jsonify(device)


@app.route('/api/devices/<device_id>', methods=['PUT'])
def update_device(device_id):
    """장치 정보 전체 수정"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

    data = request.get_json()
    data['id'] = device_id  # ID 유지
    data['updated_at'] = datetime.now().isoformat()

    if USE_SQLITE:
        cursor.execute("""
            UPDATE devices
            SET name = ?, type = ?, status = ?
            WHERE id = ?
        """, (data.get('name'), data.get('type'),
              data.get('status'), device_id))
        conn.commit()
        conn.close()

        # 업데이트된 장치 조회
        return get_device(device_id)
    else:
        memory_store['devices'][device_id] = data
        return jsonify(data)


@app.route('/api/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """장치 삭제"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404

        cursor.execute("DELETE FROM devices WHERE id = ?", (device_id,))
        conn.commit()
        conn.close()
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

        del memory_store['devices'][device_id]

    return '', 204


@app.route('/api/devices/<device_id>/command', methods=['POST'])
def send_device_command(device_id):
    """장치에 명령 전송 (시뮬레이션)"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404
        conn.close()
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    if 'command' not in data:
        return jsonify({"error": "Command required"}), 400

    # 명령 생성 (실제로는 MQTT 발행 등)
    command = {
        "device_id": device_id,
        "command": data['command'],
        "params": data.get('params', {}),
        "sent_at": datetime.now().isoformat()
    }

    # 시뮬레이션: 명령 출력
    print(f"[명령 전송] {device_id}: {command['command']}")

    return jsonify({
        "status": "sent",
        "command": command
    }), 202


# === 메인 실행 ===

if __name__ == "__main__":
    # SQLite 모드인 경우 데이터베이스 초기화
    if USE_SQLITE:
        init_db()
        print(f"데이터베이스 초기화: {DB_PATH}")
    else:
        print("메모리 저장소 모드 (시뮬레이션)")

    print("\n=== IoT REST API 서버 시작 ===")
    print("엔드포인트: http://localhost:5000/")
    print("API 문서: http://localhost:5000/")
    print("\n종료: Ctrl+C")

    # Flask 서버 시작
    app.run(host='0.0.0.0', port=5000, debug=True)
