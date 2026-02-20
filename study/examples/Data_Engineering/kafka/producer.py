"""
Kafka Producer 예제

Kafka 토픽에 메시지를 발행하는 Producer 예제입니다.

필수 패키지: pip install confluent-kafka

실행 전 Kafka 실행 필요:
  docker run -d --name kafka -p 9092:9092 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    confluentinc/cp-kafka:latest

실행: python producer.py
"""

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import json
import time
import random
from datetime import datetime
from typing import Optional


class KafkaProducerExample:
    """Kafka Producer 예제 클래스"""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'example-producer',
            'acks': 'all',  # 모든 replica 확인
            'retries': 3,
            'retry.backoff.ms': 100,
            'linger.ms': 5,  # 배치 대기 시간
            'batch.size': 16384,  # 배치 크기
        }
        self.producer = Producer(self.config)
        self.message_count = 0
        self.error_count = 0

    def delivery_callback(self, err, msg):
        """메시지 전송 결과 콜백"""
        if err:
            print(f'[ERROR] Message delivery failed: {err}')
            self.error_count += 1
        else:
            self.message_count += 1
            if self.message_count % 100 == 0:
                print(f'[INFO] Delivered {self.message_count} messages')

    def create_topic(self, topic_name: str, num_partitions: int = 3, replication_factor: int = 1):
        """토픽 생성"""
        admin_client = AdminClient({'bootstrap.servers': self.config['bootstrap.servers']})

        topic = NewTopic(
            topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )

        try:
            futures = admin_client.create_topics([topic])
            futures[topic_name].result()  # 생성 완료 대기
            print(f'[INFO] Topic "{topic_name}" created')
        except Exception as e:
            if 'already exists' in str(e):
                print(f'[INFO] Topic "{topic_name}" already exists')
            else:
                raise e

    def send_message(self, topic: str, key: Optional[str], value: dict):
        """단일 메시지 전송"""
        try:
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=json.dumps(value).encode('utf-8'),
                callback=self.delivery_callback
            )
            # 주기적으로 이벤트 처리
            self.producer.poll(0)
        except BufferError:
            # 버퍼가 꽉 찬 경우 대기
            print('[WARN] Buffer full, waiting...')
            self.producer.flush()
            self.send_message(topic, key, value)

    def flush(self):
        """모든 메시지 전송 완료 대기"""
        self.producer.flush()

    def close(self):
        """Producer 종료"""
        self.flush()
        print(f'\n[SUMMARY] Total sent: {self.message_count}, Errors: {self.error_count}')


def generate_order_event() -> dict:
    """주문 이벤트 생성"""
    products = ['laptop', 'phone', 'tablet', 'headphones', 'keyboard', 'mouse']
    statuses = ['created', 'confirmed', 'shipped', 'delivered']

    return {
        'event_type': 'order',
        'order_id': f'ORD-{random.randint(10000, 99999)}',
        'customer_id': f'CUST-{random.randint(1, 1000)}',
        'product': random.choice(products),
        'quantity': random.randint(1, 5),
        'amount': round(random.uniform(10, 1000), 2),
        'status': random.choice(statuses),
        'timestamp': datetime.now().isoformat(),
    }


def generate_clickstream_event() -> dict:
    """클릭스트림 이벤트 생성"""
    pages = ['/home', '/products', '/cart', '/checkout', '/profile', '/search']
    actions = ['view', 'click', 'scroll', 'hover']

    return {
        'event_type': 'clickstream',
        'event_id': f'EVT-{random.randint(100000, 999999)}',
        'user_id': f'USER-{random.randint(1, 500)}',
        'session_id': f'SESS-{random.randint(1000, 9999)}',
        'page': random.choice(pages),
        'action': random.choice(actions),
        'timestamp': datetime.now().isoformat(),
    }


def generate_inventory_event() -> dict:
    """재고 이벤트 생성"""
    products = ['SKU-001', 'SKU-002', 'SKU-003', 'SKU-004', 'SKU-005']
    warehouses = ['WH-EAST', 'WH-WEST', 'WH-CENTRAL']

    return {
        'event_type': 'inventory',
        'product_sku': random.choice(products),
        'warehouse': random.choice(warehouses),
        'quantity_change': random.randint(-50, 100),
        'current_stock': random.randint(0, 500),
        'timestamp': datetime.now().isoformat(),
    }


def demo_simple_producer():
    """간단한 Producer 데모"""
    print("=" * 60)
    print("Simple Producer Demo")
    print("=" * 60)

    producer = KafkaProducerExample()

    # 토픽 생성
    producer.create_topic('demo-topic')

    # 메시지 전송
    for i in range(10):
        message = {
            'id': i,
            'message': f'Hello Kafka #{i}',
            'timestamp': datetime.now().isoformat()
        }
        producer.send_message(
            topic='demo-topic',
            key=f'key-{i % 3}',  # 3개 파티션에 분배
            value=message
        )
        print(f'Sent: {message}')

    producer.close()


def demo_event_stream():
    """이벤트 스트림 데모"""
    print("\n" + "=" * 60)
    print("Event Stream Demo")
    print("=" * 60)

    producer = KafkaProducerExample()

    # 토픽 생성
    producer.create_topic('orders', num_partitions=3)
    producer.create_topic('clickstream', num_partitions=6)
    producer.create_topic('inventory', num_partitions=3)

    print("\nStreaming events (Press Ctrl+C to stop)...")

    try:
        event_count = 0
        while True:
            # 주문 이벤트 (낮은 빈도)
            if random.random() < 0.3:
                event = generate_order_event()
                producer.send_message('orders', event['customer_id'], event)

            # 클릭스트림 이벤트 (높은 빈도)
            for _ in range(random.randint(1, 5)):
                event = generate_clickstream_event()
                producer.send_message('clickstream', event['user_id'], event)

            # 재고 이벤트 (중간 빈도)
            if random.random() < 0.5:
                event = generate_inventory_event()
                producer.send_message('inventory', event['product_sku'], event)

            event_count += 1
            if event_count % 50 == 0:
                print(f'Generated {event_count} event batches')

            time.sleep(0.1)  # 100ms 간격

    except KeyboardInterrupt:
        print("\nStopping...")

    producer.close()


def demo_batch_producer():
    """배치 Producer 데모"""
    print("\n" + "=" * 60)
    print("Batch Producer Demo")
    print("=" * 60)

    producer = KafkaProducerExample()
    producer.create_topic('batch-orders')

    batch_size = 1000
    print(f"\nSending {batch_size} messages in batch...")

    start_time = time.time()

    for i in range(batch_size):
        event = generate_order_event()
        producer.send_message('batch-orders', event['order_id'], event)

    producer.flush()
    elapsed = time.time() - start_time

    print(f"\nBatch completed:")
    print(f"  Messages: {batch_size}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Throughput: {batch_size / elapsed:.0f} messages/second")

    producer.close()


def main():
    print("Kafka Producer Examples")
    print("=" * 60)
    print("Make sure Kafka is running on localhost:9092")
    print()

    # 데모 선택
    demos = {
        '1': ('Simple Producer', demo_simple_producer),
        '2': ('Event Stream', demo_event_stream),
        '3': ('Batch Producer', demo_batch_producer),
    }

    print("Available demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect demo (1-3) or 'all': ").strip()

    if choice == 'all':
        for name, func in demos.values():
            func()
    elif choice in demos:
        demos[choice][1]()
    else:
        print("Invalid choice, running simple demo...")
        demo_simple_producer()


if __name__ == "__main__":
    main()
