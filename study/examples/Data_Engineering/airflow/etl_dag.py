"""
Airflow ETL 파이프라인 DAG 예제

이 DAG는 실제 ETL 파이프라인의 구조를 보여줍니다:
- Extract: 데이터 소스에서 추출
- Transform: 데이터 정제 및 변환
- Load: 목적지에 적재
- Quality Check: 데이터 품질 검증

실행: airflow dags test etl_pipeline 2024-01-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
import json
import os


default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


# ============================================
# Extract 함수들
# ============================================
def extract_orders(**context):
    """주문 데이터 추출"""
    ds = context['ds']
    print(f"Extracting orders for {ds}")

    # 시뮬레이션: 실제로는 DB 쿼리
    orders = [
        {'order_id': 1, 'customer_id': 101, 'amount': 150.00, 'status': 'completed'},
        {'order_id': 2, 'customer_id': 102, 'amount': 250.50, 'status': 'completed'},
        {'order_id': 3, 'customer_id': 101, 'amount': 75.25, 'status': 'pending'},
        {'order_id': 4, 'customer_id': 103, 'amount': 320.00, 'status': 'completed'},
        {'order_id': 5, 'customer_id': 102, 'amount': 99.99, 'status': 'cancelled'},
    ]

    # XCom으로 데이터 전달
    context['ti'].xcom_push(key='raw_orders', value=orders)
    print(f"Extracted {len(orders)} orders")
    return len(orders)


def extract_customers(**context):
    """고객 데이터 추출"""
    customers = [
        {'customer_id': 101, 'name': 'Alice', 'segment': 'Gold'},
        {'customer_id': 102, 'name': 'Bob', 'segment': 'Silver'},
        {'customer_id': 103, 'name': 'Charlie', 'segment': 'Bronze'},
    ]

    context['ti'].xcom_push(key='raw_customers', value=customers)
    print(f"Extracted {len(customers)} customers")
    return len(customers)


# ============================================
# Transform 함수들
# ============================================
def transform_orders(**context):
    """주문 데이터 변환"""
    ti = context['ti']
    orders = ti.xcom_pull(task_ids='extract.extract_orders', key='raw_orders')

    # 변환 로직
    transformed = []
    for order in orders:
        # completed 주문만 포함
        if order['status'] == 'completed':
            transformed.append({
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'amount': order['amount'],
                'order_date': context['ds'],
            })

    ti.xcom_push(key='transformed_orders', value=transformed)
    print(f"Transformed {len(transformed)} orders (from {len(orders)})")
    return len(transformed)


def enrich_orders(**context):
    """주문 데이터에 고객 정보 추가"""
    ti = context['ti']
    orders = ti.xcom_pull(task_ids='transform.transform_orders', key='transformed_orders')
    customers = ti.xcom_pull(task_ids='extract.extract_customers', key='raw_customers')

    # 고객 정보 매핑
    customer_map = {c['customer_id']: c for c in customers}

    enriched = []
    for order in orders:
        customer = customer_map.get(order['customer_id'], {})
        enriched.append({
            **order,
            'customer_name': customer.get('name', 'Unknown'),
            'customer_segment': customer.get('segment', 'Unknown'),
        })

    ti.xcom_push(key='enriched_orders', value=enriched)
    print(f"Enriched {len(enriched)} orders")
    return enriched


# ============================================
# Load 함수들
# ============================================
def load_to_warehouse(**context):
    """데이터 웨어하우스에 적재"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    # 시뮬레이션: 실제로는 DB INSERT
    print(f"Loading {len(enriched_orders)} records to warehouse")
    for order in enriched_orders:
        print(f"  INSERT: {order}")

    return len(enriched_orders)


# ============================================
# Quality Check 함수들
# ============================================
def check_row_count(**context):
    """행 수 검증"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    row_count = len(enriched_orders)
    print(f"Row count check: {row_count}")

    if row_count == 0:
        raise ValueError("No data to load!")

    ti.xcom_push(key='row_count', value=row_count)
    return row_count


def check_data_quality(**context):
    """데이터 품질 검증"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    errors = []

    for order in enriched_orders:
        # NULL 체크
        if order.get('order_id') is None:
            errors.append(f"Missing order_id")

        # 값 범위 체크
        if order.get('amount', 0) < 0:
            errors.append(f"Negative amount: {order['order_id']}")

    if errors:
        print(f"Quality issues found: {errors}")
        ti.xcom_push(key='quality_issues', value=errors)
        return 'has_issues'
    else:
        print("Quality check passed")
        return 'no_issues'


def decide_next_step(**context):
    """품질 결과에 따른 분기"""
    ti = context['ti']
    quality_result = ti.xcom_pull(task_ids='quality.check_data_quality')

    if quality_result == 'has_issues':
        return 'quality.handle_issues'
    else:
        return 'load'


def handle_quality_issues(**context):
    """품질 이슈 처리"""
    ti = context['ti']
    issues = ti.xcom_pull(task_ids='quality.check_data_quality', key='quality_issues')
    print(f"Handling quality issues: {issues}")
    # 실제로는 알림 발송, 로그 기록 등


# ============================================
# Notification 함수
# ============================================
def send_success_notification(**context):
    """성공 알림"""
    ti = context['ti']
    row_count = ti.xcom_pull(task_ids='quality.check_row_count', key='row_count')

    message = f"""
    ETL Pipeline Completed Successfully!
    Date: {context['ds']}
    Records Loaded: {row_count}
    """
    print(message)
    # 실제로는 Slack, Email 등으로 알림


# DAG 정의
with DAG(
    dag_id='etl_pipeline',
    default_args=default_args,
    description='ETL 파이프라인 예제',
    schedule_interval='0 6 * * *',  # 매일 오전 6시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Extract TaskGroup
    with TaskGroup(group_id='extract') as extract_group:
        extract_orders_task = PythonOperator(
            task_id='extract_orders',
            python_callable=extract_orders,
        )
        extract_customers_task = PythonOperator(
            task_id='extract_customers',
            python_callable=extract_customers,
        )

    # Transform TaskGroup
    with TaskGroup(group_id='transform') as transform_group:
        transform_task = PythonOperator(
            task_id='transform_orders',
            python_callable=transform_orders,
        )
        enrich_task = PythonOperator(
            task_id='enrich_orders',
            python_callable=enrich_orders,
        )
        transform_task >> enrich_task

    # Quality TaskGroup
    with TaskGroup(group_id='quality') as quality_group:
        row_count_task = PythonOperator(
            task_id='check_row_count',
            python_callable=check_row_count,
        )
        quality_task = PythonOperator(
            task_id='check_data_quality',
            python_callable=check_data_quality,
        )
        handle_issues_task = PythonOperator(
            task_id='handle_issues',
            python_callable=handle_quality_issues,
            trigger_rule=TriggerRule.NONE_FAILED,
        )
        [row_count_task, quality_task]

    # Branch
    branch_task = BranchPythonOperator(
        task_id='branch_on_quality',
        python_callable=decide_next_step,
    )

    # Load
    load_task = PythonOperator(
        task_id='load',
        python_callable=load_to_warehouse,
    )

    # Notify
    notify_task = PythonOperator(
        task_id='notify',
        python_callable=send_success_notification,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Task 의존성
    start >> extract_group >> transform_group >> quality_group >> branch_task
    branch_task >> [load_task, handle_issues_task]
    [load_task, handle_issues_task] >> notify_task >> end


if __name__ == "__main__":
    dag.test()
