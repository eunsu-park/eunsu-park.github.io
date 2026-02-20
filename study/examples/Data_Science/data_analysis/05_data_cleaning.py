"""
데이터 전처리 (Data Cleaning/Preprocessing)
Data Cleaning and Preprocessing Techniques

실제 데이터 분석에서 가장 중요한 전처리 기법을 다룹니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# =============================================================================
# 1. 결측치 처리
# =============================================================================
def handle_missing_values():
    """결측치 탐지 및 처리"""
    print("\n[1] 결측치 처리")
    print("=" * 50)

    # 결측치가 있는 데이터 생성
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': ['a', None, 'c', 'd', np.nan]
    })

    print("원본 데이터:")
    print(df)
    print()

    # 결측치 탐지
    print("결측치 개수:")
    print(df.isnull().sum())
    print(f"\n결측치 비율:\n{df.isnull().mean() * 100}")

    # 처리 방법들
    print("\n--- 결측치 처리 방법 ---")

    # 1. 행 삭제
    df_dropna = df.dropna()
    print(f"\n1. 행 삭제 (dropna):\n{df_dropna}")

    # 2. 특정 열에서만 삭제
    df_drop_subset = df.dropna(subset=['A', 'C'])
    print(f"\n2. A, C 열 기준 삭제:\n{df_drop_subset}")

    # 3. 값으로 채우기
    df_fillna = df.copy()
    df_fillna['A'] = df_fillna['A'].fillna(df_fillna['A'].mean())
    df_fillna['B'] = df_fillna['B'].fillna(df_fillna['B'].median())
    print(f"\n3. 평균/중앙값으로 채우기:\n{df_fillna}")

    # 4. 전방/후방 채우기
    df_ffill = df.fillna(method='ffill')
    print(f"\n4. 전방 채우기 (ffill):\n{df_ffill}")

    # 5. 보간법
    df_interpolate = df.copy()
    df_interpolate['A'] = df_interpolate['A'].interpolate()
    df_interpolate['B'] = df_interpolate['B'].interpolate()
    print(f"\n5. 보간법 (interpolate):\n{df_interpolate}")


# =============================================================================
# 2. 이상치 탐지 및 처리
# =============================================================================
def handle_outliers():
    """이상치 탐지 및 처리"""
    print("\n[2] 이상치 탐지 및 처리")
    print("=" * 50)

    np.random.seed(42)

    # 이상치가 포함된 데이터
    normal_data = np.random.normal(100, 10, 100)
    outliers = np.array([200, -50, 250])
    data = np.concatenate([normal_data, outliers])
    np.random.shuffle(data)

    df = pd.DataFrame({'value': data})

    print(f"데이터 크기: {len(df)}")
    print(f"평균: {df['value'].mean():.2f}")
    print(f"표준편차: {df['value'].std():.2f}")

    # 방법 1: IQR 방법
    print("\n--- IQR 방법 ---")
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"정상 범위: [{lower_bound:.2f}, {upper_bound:.2f}]")

    outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"이상치 개수: {len(outliers_iqr)}")
    print(f"이상치 값: {outliers_iqr['value'].values}")

    # 방법 2: Z-score 방법
    print("\n--- Z-score 방법 ---")
    z_scores = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
    outliers_z = df[z_scores > 3]
    print(f"이상치 개수 (|z| > 3): {len(outliers_z)}")

    # 이상치 처리
    print("\n--- 이상치 처리 ---")

    # 1. 제거
    df_no_outliers = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    print(f"1. 제거 후 크기: {len(df_no_outliers)}")

    # 2. 경계값으로 대체 (Winsorizing)
    df_winsorized = df.copy()
    df_winsorized['value'] = df_winsorized['value'].clip(lower_bound, upper_bound)
    print(f"2. Winsorizing 후 최대값: {df_winsorized['value'].max():.2f}")

    # 3. 중앙값으로 대체
    df_median = df.copy()
    median_val = df['value'].median()
    df_median.loc[(df['value'] < lower_bound) | (df['value'] > upper_bound), 'value'] = median_val
    print(f"3. 중앙값 대체 후 평균: {df_median['value'].mean():.2f}")


# =============================================================================
# 3. 데이터 타입 변환
# =============================================================================
def data_type_conversion():
    """데이터 타입 변환"""
    print("\n[3] 데이터 타입 변환")
    print("=" * 50)

    df = pd.DataFrame({
        'int_col': ['1', '2', '3', '4', '5'],
        'float_col': ['1.1', '2.2', '3.3', '4.4', '5.5'],
        'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'bool_col': ['True', 'False', 'True', 'False', 'True'],
        'cat_col': ['A', 'B', 'A', 'C', 'B']
    })

    print("원본 데이터 타입:")
    print(df.dtypes)
    print()

    # 타입 변환
    df['int_col'] = df['int_col'].astype(int)
    df['float_col'] = df['float_col'].astype(float)
    df['date_col'] = pd.to_datetime(df['date_col'])
    df['bool_col'] = df['bool_col'].map({'True': True, 'False': False})
    df['cat_col'] = df['cat_col'].astype('category')

    print("변환 후 데이터 타입:")
    print(df.dtypes)
    print()

    print("변환된 데이터:")
    print(df)

    # 메모리 사용량 비교
    print(f"\n카테고리 타입 메모리 절약:")
    print(f"  object 타입: {df['cat_col'].astype('object').memory_usage()} bytes")
    print(f"  category 타입: {df['cat_col'].memory_usage()} bytes")


# =============================================================================
# 4. 중복 데이터 처리
# =============================================================================
def handle_duplicates():
    """중복 데이터 처리"""
    print("\n[4] 중복 데이터 처리")
    print("=" * 50)

    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David'],
        'age': [25, 30, 25, 35, 30, 40],
        'city': ['Seoul', 'Busan', 'Seoul', 'Daegu', 'Busan', 'Seoul']
    })

    print("원본 데이터:")
    print(df)

    # 중복 확인
    print(f"\n중복 행 수: {df.duplicated().sum()}")
    print("중복된 행:")
    print(df[df.duplicated()])

    # 특정 열 기준 중복
    print(f"\n'name' 기준 중복 수: {df.duplicated(subset=['name']).sum()}")

    # 중복 제거
    df_unique = df.drop_duplicates()
    print(f"\n중복 제거 후:\n{df_unique}")

    df_unique_name = df.drop_duplicates(subset=['name'], keep='first')
    print(f"\n'name' 기준 중복 제거 (첫 번째 유지):\n{df_unique_name}")


# =============================================================================
# 5. 정규화와 표준화
# =============================================================================
def normalization_standardization():
    """정규화와 표준화"""
    print("\n[5] 정규화와 표준화")
    print("=" * 50)

    np.random.seed(42)

    df = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 10),
        'feature2': np.random.normal(50, 5, 10),
        'feature3': np.random.exponential(10, 10)
    })

    print("원본 데이터 통계:")
    print(df.describe().round(2))

    # 1. Min-Max 정규화 (0-1 스케일링)
    df_minmax = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_minmax[col] = (df[col] - min_val) / (max_val - min_val)

    print("\n1. Min-Max 정규화 (0-1):")
    print(df_minmax.describe().round(4))

    # 2. Z-score 표준화
    df_zscore = df.copy()
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df_zscore[col] = (df[col] - mean_val) / std_val

    print("\n2. Z-score 표준화:")
    print(df_zscore.describe().round(4))

    # 3. Robust 스케일링 (이상치에 강건)
    df_robust = df.copy()
    for col in df.columns:
        median_val = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        df_robust[col] = (df[col] - median_val) / iqr

    print("\n3. Robust 스케일링 (IQR 기반):")
    print(df_robust.describe().round(4))


# =============================================================================
# 6. 범주형 변수 인코딩
# =============================================================================
def categorical_encoding():
    """범주형 변수 인코딩"""
    print("\n[6] 범주형 변수 인코딩")
    print("=" * 50)

    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue', 'red'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'price': [100, 150, 200, 150, 100]
    })

    print("원본 데이터:")
    print(df)

    # 1. 라벨 인코딩
    print("\n1. 라벨 인코딩:")
    df_label = df.copy()
    df_label['color_encoded'] = df_label['color'].astype('category').cat.codes
    df_label['size_encoded'] = df_label['size'].map({'S': 0, 'M': 1, 'L': 2})
    print(df_label)

    # 2. 원-핫 인코딩
    print("\n2. 원-핫 인코딩:")
    df_onehot = pd.get_dummies(df, columns=['color', 'size'])
    print(df_onehot)

    # 3. 빈도 인코딩
    print("\n3. 빈도 인코딩:")
    df_freq = df.copy()
    freq_map = df['color'].value_counts() / len(df)
    df_freq['color_freq'] = df_freq['color'].map(freq_map)
    print(df_freq)


# =============================================================================
# 7. 문자열 처리
# =============================================================================
def string_processing():
    """문자열 처리"""
    print("\n[7] 문자열 처리")
    print("=" * 50)

    df = pd.DataFrame({
        'name': ['  John Doe  ', 'jane smith', 'BOB JONES', 'Alice Brown'],
        'email': ['john@example.com', 'jane@EXAMPLE.COM', 'bob@Example.com', 'alice@example.com'],
        'phone': ['010-1234-5678', '01098765432', '010 1111 2222', '010.3333.4444']
    })

    print("원본 데이터:")
    print(df)

    # 문자열 처리
    df_clean = df.copy()

    # 공백 제거 및 대소문자 정리
    df_clean['name'] = df_clean['name'].str.strip().str.title()

    # 소문자 변환
    df_clean['email'] = df_clean['email'].str.lower()

    # 전화번호 정규화
    df_clean['phone'] = df_clean['phone'].str.replace(r'[^0-9]', '', regex=True)

    print("\n정리된 데이터:")
    print(df_clean)

    # 문자열 추출
    print("\n문자열 분리:")
    df_clean[['first_name', 'last_name']] = df_clean['name'].str.split(' ', n=1, expand=True)
    print(df_clean[['name', 'first_name', 'last_name']])


# =============================================================================
# 8. 날짜/시간 처리
# =============================================================================
def datetime_processing():
    """날짜/시간 처리"""
    print("\n[8] 날짜/시간 처리")
    print("=" * 50)

    df = pd.DataFrame({
        'date_str': ['2024-01-15', '2024/02/20', '15-Mar-2024', '2024.04.10'],
        'timestamp': pd.date_range('2024-01-01', periods=4, freq='ME'),
        'value': [100, 150, 120, 180]
    })

    print("원본 데이터:")
    print(df)

    # 날짜 파싱
    df['date_parsed'] = pd.to_datetime(df['date_str'])

    # 날짜 요소 추출
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.day_name()
    df['quarter'] = df['timestamp'].dt.quarter

    print("\n날짜 요소 추출:")
    print(df[['timestamp', 'year', 'month', 'day', 'weekday', 'quarter']])

    # 날짜 연산
    df['days_since'] = (pd.Timestamp('2024-12-31') - df['timestamp']).dt.days

    print("\n날짜 연산 (2024-12-31까지 남은 일수):")
    print(df[['timestamp', 'days_since']])


# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("데이터 전처리 예제")
    print("=" * 60)

    handle_missing_values()
    handle_outliers()
    data_type_conversion()
    handle_duplicates()
    normalization_standardization()
    categorical_encoding()
    string_processing()
    datetime_processing()

    print("\n" + "=" * 60)
    print("데이터 전처리 체크리스트")
    print("=" * 60)
    print("""
    1. 데이터 로드 및 확인
       - head(), info(), describe()
       - shape, dtypes

    2. 결측치 처리
       - isnull().sum() 으로 확인
       - 삭제 또는 대체 (평균, 중앙값, 최빈값, 보간)

    3. 이상치 처리
       - IQR 또는 Z-score로 탐지
       - 제거, 경계값 대체, 또는 변환

    4. 데이터 타입 변환
       - 숫자, 날짜, 범주형으로 적절히 변환
       - category 타입으로 메모리 절약

    5. 중복 제거
       - duplicated() 확인
       - drop_duplicates()

    6. 스케일링/정규화
       - Min-Max: 범위가 중요할 때
       - Z-score: 분포가 중요할 때
       - Robust: 이상치가 있을 때

    7. 범주형 인코딩
       - 라벨 인코딩: 순서가 있는 변수
       - 원-핫 인코딩: 순서가 없는 변수

    8. 문자열/날짜 정리
       - 공백 제거, 대소문자 통일
       - 날짜 파싱 및 요소 추출
    """)


if __name__ == "__main__":
    main()
