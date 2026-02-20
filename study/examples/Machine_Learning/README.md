# Machine Learning 예제

Machine_Learning 폴더의 14개 레슨에 해당하는 실행 가능한 Jupyter Notebook 예제입니다.

## 폴더 구조

```
examples/
├── 01_linear_regression.ipynb      # 선형 회귀
├── 02_logistic_regression.ipynb    # 로지스틱 회귀
├── 03_model_evaluation.ipynb       # 모델 평가 지표
├── 04_cross_validation.ipynb       # 교차 검증
├── 05_preprocessing.ipynb          # 데이터 전처리
├── 06_decision_tree.ipynb          # 결정 트리
├── 07_random_forest.ipynb          # 랜덤 포레스트
├── 08_xgboost_lightgbm.ipynb       # XGBoost, LightGBM
├── 09_svm.ipynb                    # SVM (Support Vector Machine)
├── 10_knn_naive_bayes.ipynb        # k-NN, 나이브 베이즈
├── 11_clustering.ipynb             # K-Means, DBSCAN
├── 12_pca.ipynb                    # PCA, t-SNE 차원 축소
├── 13_pipeline.ipynb               # sklearn 파이프라인
├── 14_kaggle_project.ipynb         # 실전 Kaggle 프로젝트
├── datasets/                       # 예제 데이터셋
└── README.md
```

## 실행 방법

### 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv ml-env
source ml-env/bin/activate  # Windows: ml-env\Scripts\activate

# 필요한 패키지 설치
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# XGBoost, LightGBM (08 레슨용)
pip install xgboost lightgbm
```

### Jupyter Notebook 실행

```bash
cd Machine_Learning/examples
jupyter notebook

# 또는 JupyterLab
jupyter lab
```

## 레슨별 예제 목록

| 레슨 | 주제 | 핵심 내용 |
|------|------|----------|
| 01 | 선형 회귀 | 단순/다중 회귀, MSE, R² |
| 02 | 로지스틱 회귀 | 이진/다중 분류, ROC-AUC |
| 03 | 모델 평가 | 정확도, 정밀도, 재현율, F1 |
| 04 | 교차 검증 | K-Fold, Stratified, GridSearchCV |
| 05 | 전처리 | 스케일링, 인코딩, 결측치 |
| 06 | 결정 트리 | 트리 시각화, 과적합 방지 |
| 07 | 랜덤 포레스트 | 배깅, OOB, 특성 중요도 |
| 08 | XGBoost/LightGBM | 그래디언트 부스팅, 조기 종료 |
| 09 | SVM | 커널 트릭, 하이퍼플레인 |
| 10 | k-NN/나이브 베이즈 | 거리 기반, 확률 기반 분류 |
| 11 | 클러스터링 | K-Means, DBSCAN, 실루엣 |
| 12 | 차원 축소 | PCA, t-SNE, 설명 분산 |
| 13 | 파이프라인 | Pipeline, ColumnTransformer |
| 14 | Kaggle 프로젝트 | Titanic, 특성 공학 |

## 학습 순서

1. **기초**: 01 → 02 → 03 → 04 → 05
2. **트리 모델**: 06 → 07 → 08
3. **기타 알고리즘**: 09 → 10
4. **비지도 학습**: 11 → 12
5. **실전**: 13 → 14

## 데이터셋

예제에서 사용하는 데이터셋:

| 데이터셋 | 출처 | 용도 |
|---------|------|------|
| Iris | sklearn | 분류 (다중 클래스) |
| Wine | sklearn | 분류 (다중 클래스) |
| California Housing | sklearn | 회귀 |
| Digits | sklearn | 분류 (이미지) |
| Titanic | Kaggle | 분류 (실전) |

## 필요 패키지

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
xgboost>=1.5.0      # 08 레슨
lightgbm>=3.3.0     # 08 레슨
```

## 참고 자료

- [scikit-learn 공식 문서](https://scikit-learn.org/stable/)
- [Kaggle](https://www.kaggle.com/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
