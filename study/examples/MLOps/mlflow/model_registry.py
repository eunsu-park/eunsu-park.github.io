"""
MLflow Model Registry Example
=============================

MLflow Model Registry를 사용한 모델 버전 관리 예제입니다.

실행 방법:
    # 먼저 tracking_example.py를 실행하여 모델을 학습/저장한 후
    python model_registry.py
"""

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# MLflow 설정
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "iris-classifier"


def setup():
    """MLflow 설정"""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("model-registry-demo")
    return MlflowClient()


def train_and_register_model(client, version_tag: str):
    """모델 학습 및 레지스트리 등록"""
    # 데이터 준비
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"training-{version_tag}") as run:
        # 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 평가
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        # 모델 저장 및 등록
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        print(f"\n모델 등록 완료: {MODEL_NAME}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Accuracy: {accuracy:.4f}")

        return run.info.run_id


def get_model_versions(client):
    """등록된 모델 버전 조회"""
    print(f"\n{'='*50}")
    print(f"모델 '{MODEL_NAME}' 버전 목록:")
    print("="*50)

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for v in versions:
            print(f"\n버전 {v.version}:")
            print(f"  상태: {v.current_stage}")
            print(f"  Run ID: {v.run_id}")
            print(f"  생성일: {v.creation_timestamp}")
            if v.description:
                print(f"  설명: {v.description}")
        return versions
    except Exception as e:
        print(f"모델을 찾을 수 없습니다: {e}")
        return []


def transition_to_staging(client, version: str):
    """모델을 Staging으로 전환"""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"\n모델 v{version}을 Staging으로 전환했습니다.")


def transition_to_production(client, version: str):
    """모델을 Production으로 전환"""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"\n모델 v{version}을 Production으로 전환했습니다.")


def update_model_description(client, version: str, description: str):
    """모델 설명 업데이트"""
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=description
    )
    print(f"\n모델 v{version} 설명을 업데이트했습니다.")


def add_model_tag(client, version: str, key: str, value: str):
    """모델 태그 추가"""
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key=key,
        value=value
    )
    print(f"\n모델 v{version}에 태그 추가: {key}={value}")


def load_model_by_stage(stage: str):
    """스테이지별 모델 로드"""
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{stage}")
        print(f"\n{stage} 모델 로드 성공!")
        return model
    except Exception as e:
        print(f"\n{stage} 모델 로드 실패: {e}")
        return None


def demo_workflow(client):
    """전체 워크플로우 데모"""
    print("\n" + "="*60)
    print("MLflow Model Registry 워크플로우 데모")
    print("="*60)

    # 1. 첫 번째 모델 등록
    print("\n[1] 첫 번째 모델 학습 및 등록...")
    train_and_register_model(client, "v1")

    # 2. 버전 조회
    versions = get_model_versions(client)
    if not versions:
        return

    latest_version = max(v.version for v in versions)

    # 3. 설명 추가
    print("\n[2] 모델 설명 추가...")
    update_model_description(
        client, latest_version,
        "Initial model trained on Iris dataset with Random Forest"
    )

    # 4. 태그 추가
    print("\n[3] 모델 태그 추가...")
    add_model_tag(client, latest_version, "validated", "true")
    add_model_tag(client, latest_version, "dataset", "iris")

    # 5. Staging 전환
    print("\n[4] Staging으로 전환...")
    transition_to_staging(client, latest_version)

    # 6. 두 번째 모델 등록
    print("\n[5] 두 번째 모델 학습 및 등록 (개선 버전)...")
    train_and_register_model(client, "v2")

    # 7. 버전 재조회
    versions = get_model_versions(client)
    new_latest = max(v.version for v in versions)

    # 8. 새 버전을 Staging으로
    print("\n[6] 새 버전을 Staging으로...")
    transition_to_staging(client, new_latest)

    # 9. Production 승격
    print("\n[7] Production으로 승격...")
    transition_to_production(client, new_latest)

    # 10. 최종 상태 확인
    print("\n[8] 최종 모델 상태:")
    get_model_versions(client)

    # 11. Production 모델 로드 테스트
    print("\n[9] Production 모델 로드 테스트...")
    model = load_model_by_stage("Production")
    if model:
        # 간단한 예측 테스트
        iris = load_iris()
        sample = iris.data[:3]
        predictions = model.predict(sample)
        print(f"  샘플 예측 결과: {predictions}")
        print(f"  실제 레이블: {iris.target[:3]}")


def main():
    """메인 함수"""
    client = setup()

    print("\nMLflow Model Registry 예제")
    print("="*50)
    print("\n옵션:")
    print("1. 새 모델 학습 및 등록")
    print("2. 등록된 모델 조회")
    print("3. 전체 워크플로우 데모")

    choice = input("\n선택 (1/2/3): ").strip()

    if choice == "1":
        train_and_register_model(client, "manual")
    elif choice == "2":
        get_model_versions(client)
    elif choice == "3":
        demo_workflow(client)
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
