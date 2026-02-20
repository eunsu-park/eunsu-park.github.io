"""
Airflow 기본 DAG 예제

이 DAG는 기본적인 Airflow 워크플로우를 보여줍니다:
- PythonOperator로 Python 함수 실행
- BashOperator로 쉘 명령 실행
- Task 의존성 정의

실행: airflow dags test simple_dag 2024-01-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


# 기본 인자 설정
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email': ['data-alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Python 함수 정의
def print_hello():
    """인사 메시지 출력"""
    print("Hello from Airflow!")
    return "hello_returned"


def print_date(**context):
    """실행 날짜 출력"""
    execution_date = context['ds']
    print(f"Execution date: {execution_date}")
    return execution_date


def process_data(value: int, multiplier: int = 2, **context):
    """데이터 처리 예시"""
    result = value * multiplier
    print(f"Processing: {value} * {multiplier} = {result}")

    # XCom으로 결과 저장
    context['ti'].xcom_push(key='processed_value', value=result)
    return result


def summarize(**context):
    """이전 Task 결과 요약"""
    ti = context['ti']

    # XCom에서 값 가져오기
    hello_result = ti.xcom_pull(task_ids='hello_task')
    processed_value = ti.xcom_pull(task_ids='process_task', key='processed_value')

    print(f"Summary:")
    print(f"  - Hello result: {hello_result}")
    print(f"  - Processed value: {processed_value}")
    print(f"  - Execution date: {context['ds']}")


# DAG 정의
with DAG(
    dag_id='simple_dag',
    default_args=default_args,
    description='간단한 Airflow DAG 예제',
    schedule_interval='@daily',  # 매일 실행
    start_date=datetime(2024, 1, 1),
    catchup=False,  # 과거 실행 건너뛰기
    tags=['example', 'tutorial'],
) as dag:

    # Task 정의
    start = EmptyOperator(task_id='start')

    hello_task = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello,
    )

    date_task = PythonOperator(
        task_id='date_task',
        python_callable=print_date,
    )

    bash_task = BashOperator(
        task_id='bash_task',
        bash_command='echo "Current time: $(date)" && sleep 2',
    )

    process_task = PythonOperator(
        task_id='process_task',
        python_callable=process_data,
        op_kwargs={'value': 10, 'multiplier': 5},
    )

    summary_task = PythonOperator(
        task_id='summary_task',
        python_callable=summarize,
    )

    end = EmptyOperator(task_id='end')

    # Task 의존성 정의
    #     ┌─ hello_task ─┐
    # start ─┤             ├─ process_task ─ summary_task ─ end
    #     └─ date_task ──┘
    #             └─ bash_task ──┘

    start >> [hello_task, date_task]
    hello_task >> process_task
    date_task >> [bash_task, process_task]
    [bash_task, process_task] >> summary_task >> end


if __name__ == "__main__":
    # 로컬 테스트
    dag.test()
