"""
Kafka Consumer 예제

Kafka 토픽에서 메시지를 소비하는 Consumer 예제입니다.

필수 패키지: pip install confluent-kafka

실행 전:
  1. Kafka 실행 필요
  2. producer.py로 메시지 발행

실행: python consumer.py
"""

from confluent_kafka import Consumer, KafkaError, KafkaException
import json
import signal
import sys
from datetime import datetime
from typing import Callable, Optional, Dict, List
from collections import defaultdict


class KafkaConsumerExample:
    """Kafka Consumer 예제 클래스"""

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        group_id: str = 'example-consumer-group',
        auto_commit: bool = True
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',  # 처음부터 읽기
            'enable.auto.commit': auto_commit,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 45000,
            'max.poll.interval.ms': 300000,
        }
        self.consumer = Consumer(self.config)
        self.running = True
        self.message_count = 0
        self.error_count = 0

        # Graceful shutdown 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """종료 시그널 핸들러"""
        print("\n[INFO] Shutdown signal received")
        self.running = False

    def subscribe(self, topics: List[str]):
        """토픽 구독"""
        def on_assign(consumer, partitions):
            print(f'[INFO] Partitions assigned: {[p.partition for p in partitions]}')

        def on_revoke(consumer, partitions):
            print(f'[INFO] Partitions revoked: {[p.partition for p in partitions]}')

        self.consumer.subscribe(topics, on_assign=on_assign, on_revoke=on_revoke)
        print(f'[INFO] Subscribed to topics: {topics}')

    def consume(self, handler: Callable[[dict, dict], None], timeout: float = 1.0):
        """메시지 소비 루프"""
        print('[INFO] Starting consumer loop...')

        while self.running:
            try:
                msg = self.consumer.poll(timeout=timeout)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # 파티션 끝에 도달
                        continue
                    else:
                        print(f'[ERROR] Consumer error: {msg.error()}')
                        self.error_count += 1
                        continue

                # 메시지 파싱
                try:
                    key = msg.key().decode('utf-8') if msg.key() else None
                    value = json.loads(msg.value().decode('utf-8'))

                    metadata = {
                        'topic': msg.topic(),
                        'partition': msg.partition(),
                        'offset': msg.offset(),
                        'timestamp': msg.timestamp(),
                        'key': key,
                    }

                    # 핸들러 호출
                    handler(value, metadata)
                    self.message_count += 1

                except json.JSONDecodeError as e:
                    print(f'[ERROR] Failed to parse message: {e}')
                    self.error_count += 1

            except KafkaException as e:
                print(f'[ERROR] Kafka exception: {e}')
                self.error_count += 1

        self.close()

    def commit(self):
        """수동 커밋"""
        self.consumer.commit()

    def close(self):
        """Consumer 종료"""
        print('\n[INFO] Closing consumer...')
        self.consumer.close()
        print(f'[SUMMARY] Total consumed: {self.message_count}, Errors: {self.error_count}')


class MessageAggregator:
    """메시지 집계 클래스"""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.counts = defaultdict(int)
        self.amounts = defaultdict(float)
        self.last_report = datetime.now()

    def add(self, event_type: str, amount: float = 0):
        """이벤트 추가"""
        self.counts[event_type] += 1
        self.amounts[event_type] += amount

    def should_report(self) -> bool:
        """보고 시간 확인"""
        elapsed = (datetime.now() - self.last_report).seconds
        return elapsed >= self.window_seconds

    def report_and_reset(self) -> dict:
        """보고 및 리셋"""
        report = {
            'window_end': datetime.now().isoformat(),
            'counts': dict(self.counts),
            'amounts': dict(self.amounts),
        }
        self.counts.clear()
        self.amounts.clear()
        self.last_report = datetime.now()
        return report


def demo_simple_consumer():
    """간단한 Consumer 데모"""
    print("=" * 60)
    print("Simple Consumer Demo")
    print("=" * 60)

    consumer = KafkaConsumerExample(group_id='simple-consumer-group')
    consumer.subscribe(['demo-topic'])

    def message_handler(value: dict, metadata: dict):
        print(f"Received: topic={metadata['topic']}, "
              f"partition={metadata['partition']}, "
              f"offset={metadata['offset']}")
        print(f"  Key: {metadata['key']}")
        print(f"  Value: {value}")
        print()

    consumer.consume(message_handler)


def demo_order_consumer():
    """주문 이벤트 Consumer 데모"""
    print("\n" + "=" * 60)
    print("Order Consumer Demo")
    print("=" * 60)

    consumer = KafkaConsumerExample(group_id='order-consumer-group')
    consumer.subscribe(['orders'])

    aggregator = MessageAggregator(window_seconds=10)

    def order_handler(value: dict, metadata: dict):
        # 이벤트 집계
        aggregator.add(value.get('status', 'unknown'), value.get('amount', 0))

        # 주기적 보고
        if aggregator.should_report():
            report = aggregator.report_and_reset()
            print(f"\n[REPORT] {report['window_end']}")
            print(f"  Order counts: {report['counts']}")
            print(f"  Total amounts: {report['amounts']}")

        # 특정 조건 처리 (예: 고액 주문)
        if value.get('amount', 0) > 500:
            print(f"[ALERT] High-value order: {value['order_id']} - ${value['amount']}")

    print("Processing orders (Press Ctrl+C to stop)...")
    consumer.consume(order_handler)


def demo_multi_topic_consumer():
    """멀티 토픽 Consumer 데모"""
    print("\n" + "=" * 60)
    print("Multi-Topic Consumer Demo")
    print("=" * 60)

    consumer = KafkaConsumerExample(group_id='multi-topic-group')
    consumer.subscribe(['orders', 'clickstream', 'inventory'])

    topic_counts = defaultdict(int)

    def multi_handler(value: dict, metadata: dict):
        topic = metadata['topic']
        topic_counts[topic] += 1

        # 토픽별 처리
        if topic == 'orders':
            print(f"[ORDER] {value.get('order_id')}: ${value.get('amount')}")
        elif topic == 'clickstream':
            if topic_counts[topic] % 10 == 0:  # 10개마다 출력
                print(f"[CLICK] Processed {topic_counts[topic]} events")
        elif topic == 'inventory':
            if value.get('current_stock', 100) < 10:
                print(f"[LOW STOCK] {value.get('product_sku')}: {value.get('current_stock')} units")

    print("Processing multiple topics (Press Ctrl+C to stop)...")
    consumer.consume(multi_handler)


def demo_manual_commit_consumer():
    """수동 커밋 Consumer 데모"""
    print("\n" + "=" * 60)
    print("Manual Commit Consumer Demo")
    print("=" * 60)

    consumer = KafkaConsumerExample(
        group_id='manual-commit-group',
        auto_commit=False  # 자동 커밋 비활성화
    )
    consumer.subscribe(['batch-orders'])

    batch = []
    batch_size = 10

    def batch_handler(value: dict, metadata: dict):
        batch.append(value)

        # 배치 처리
        if len(batch) >= batch_size:
            print(f"\n[BATCH] Processing {len(batch)} messages...")

            # 배치 처리 로직
            total_amount = sum(msg.get('amount', 0) for msg in batch)
            print(f"  Total amount: ${total_amount:.2f}")

            # 처리 성공 후 커밋
            consumer.commit()
            print(f"  Committed offset")

            batch.clear()

    print("Processing in batches with manual commit (Press Ctrl+C to stop)...")
    consumer.consume(batch_handler)


def demo_stateful_consumer():
    """상태 유지 Consumer 데모"""
    print("\n" + "=" * 60)
    print("Stateful Consumer Demo")
    print("=" * 60)

    consumer = KafkaConsumerExample(group_id='stateful-group')
    consumer.subscribe(['orders'])

    # 고객별 상태 유지
    customer_state = defaultdict(lambda: {
        'order_count': 0,
        'total_spent': 0.0,
        'last_order': None
    })

    def stateful_handler(value: dict, metadata: dict):
        customer_id = value.get('customer_id')
        if not customer_id:
            return

        state = customer_state[customer_id]
        state['order_count'] += 1
        state['total_spent'] += value.get('amount', 0)
        state['last_order'] = value.get('timestamp')

        # VIP 고객 감지
        if state['total_spent'] > 2000:
            print(f"[VIP] Customer {customer_id}: "
                  f"{state['order_count']} orders, ${state['total_spent']:.2f} total")

    print("Tracking customer state (Press Ctrl+C to stop)...")
    consumer.consume(stateful_handler)


def main():
    print("Kafka Consumer Examples")
    print("=" * 60)
    print("Make sure Kafka is running and has messages")
    print("Run producer.py first to generate messages")
    print()

    demos = {
        '1': ('Simple Consumer', demo_simple_consumer),
        '2': ('Order Consumer', demo_order_consumer),
        '3': ('Multi-Topic Consumer', demo_multi_topic_consumer),
        '4': ('Manual Commit Consumer', demo_manual_commit_consumer),
        '5': ('Stateful Consumer', demo_stateful_consumer),
    }

    print("Available demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect demo (1-5): ").strip()

    if choice in demos:
        demos[choice][1]()
    else:
        print("Invalid choice, running simple demo...")
        demo_simple_consumer()


if __name__ == "__main__":
    main()
