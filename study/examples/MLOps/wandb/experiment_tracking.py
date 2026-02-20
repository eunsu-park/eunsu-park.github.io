"""
Weights & Biases Experiment Tracking Example
============================================

W&B를 사용한 실험 추적 예제입니다.

실행 방법:
    # W&B 로그인
    wandb login

    # 스크립트 실행
    python experiment_tracking.py
"""

import wandb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# W&B 프로젝트 설정
PROJECT_NAME = "breast-cancer-classification"
ENTITY = None  # 팀 이름 (개인이면 None)


def load_data():
    """데이터 로드 및 분할"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names


def calculate_metrics(y_true, y_pred, y_proba=None):
    """메트릭 계산"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def train_with_wandb(model_name, model, params, X_train, X_test, y_train, y_test, feature_names):
    """W&B로 실험 추적하며 모델 학습"""

    # W&B 초기화
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=model_name,
        config=params,
        tags=["baseline", model_name.lower()],
        notes=f"Training {model_name} on breast cancer dataset"
    )

    # 추가 설정 로깅
    wandb.config.update({
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1]
    })

    # 모델 학습
    model.fit(X_train, y_train)

    # 교차 검증
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    wandb.log({
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    })

    # 학습 곡선 시뮬레이션 (일부 모델에서)
    if hasattr(model, "n_estimators"):
        # 단계별 성능 기록
        for i in range(1, params.get("n_estimators", 100) + 1, 10):
            partial_model = type(model)(**{**params, "n_estimators": i})
            partial_model.fit(X_train, y_train)
            train_score = partial_model.score(X_train, y_train)
            val_score = partial_model.score(X_test, y_test)
            wandb.log({
                "train_accuracy": train_score,
                "val_accuracy": val_score,
                "n_estimators": i
            })

    # 최종 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 메트릭 로깅
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    wandb.log(metrics)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Feature Importance (해당하는 경우)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # 상위 15개

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
        plt.close()

        # 테이블로도 로깅
        importance_data = [
            [feature_names[i], importance[i]]
            for i in indices
        ]
        table = wandb.Table(columns=["feature", "importance"], data=importance_data)
        wandb.log({"feature_importance_table": table})

    # ROC Curve (확률 예측 가능한 경우)
    if y_proba is not None:
        wandb.log({
            "roc_curve": wandb.plot.roc_curve(y_test, np.column_stack([1-y_proba, y_proba]))
        })

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    wandb.log({
        "classification_report": report
    })

    # 모델 아티팩트 저장
    artifact = wandb.Artifact(
        name=f"{model_name.lower()}-model",
        type="model",
        description=f"{model_name} trained on breast cancer dataset"
    )
    # 실제 프로덕션에서는 모델 파일 저장
    # artifact.add_file("model.pkl")
    wandb.log_artifact(artifact)

    # 결과 출력
    print(f"\n{model_name} 결과:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 실행 종료
    wandb.finish()

    return metrics


def hyperparameter_sweep():
    """하이퍼파라미터 스윕 예제"""

    # 스윕 설정
    sweep_config = {
        "name": "rf-hyperparameter-sweep",
        "method": "bayes",  # random, grid, bayes
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [50, 100, 150, 200]
            },
            "max_depth": {
                "values": [3, 5, 7, 10, None]
            },
            "min_samples_split": {
                "distribution": "int_uniform",
                "min": 2,
                "max": 20
            },
            "min_samples_leaf": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 10
            }
        }
    }

    def train_sweep():
        """스윕에서 실행될 학습 함수"""
        wandb.init()
        config = wandb.config

        X_train, X_test, y_train, y_test, _, _ = load_data()

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )

        model.fit(X_train, y_train)
        val_accuracy = model.score(X_test, y_test)

        wandb.log({"val_accuracy": val_accuracy})
        wandb.finish()

    # 스윕 생성 및 실행
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    print(f"\n스윕 ID: {sweep_id}")
    print("스윕을 실행하려면:")
    print(f"  wandb agent {sweep_id}")

    # 로컬에서 실행 (선택적)
    # wandb.agent(sweep_id, function=train_sweep, count=20)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("Weights & Biases 실험 추적 예제")
    print("="*60)

    # 데이터 로드
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    print(f"\n데이터셋:")
    print(f"  학습 데이터: {len(X_train)} 샘플")
    print(f"  테스트 데이터: {len(X_test)} 샘플")
    print(f"  피처 수: {len(feature_names)}")

    # 모델 정의
    models = [
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            {"n_estimators": 100, "max_depth": 10}
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "solver": "lbfgs"}
        )
    ]

    # 모델 학습
    results = {}
    for model_name, model, params in models:
        print(f"\n{'='*40}")
        print(f"{model_name} 학습 중...")
        metrics = train_with_wandb(
            model_name, model, params,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results[model_name] = metrics

    # 결과 요약
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} {metrics.get('roc_auc', 0):<12.4f}")

    print(f"\nW&B 대시보드에서 자세한 결과를 확인하세요:")
    print(f"  https://wandb.ai/{ENTITY or 'your-username'}/{PROJECT_NAME}")


if __name__ == "__main__":
    main()
