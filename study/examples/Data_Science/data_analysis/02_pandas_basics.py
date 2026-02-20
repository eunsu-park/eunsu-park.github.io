"""
Pandas 기초 (Pandas Basics)
Fundamental Pandas Operations

Pandas는 데이터 분석을 위한 핵심 라이브러리입니다.
"""

import pandas as pd
import numpy as np


# =============================================================================
# 1. DataFrame과 Series 생성
# =============================================================================
def create_dataframe():
    """DataFrame과 Series 생성"""
    print("\n[1] DataFrame과 Series 생성")
    print("=" * 50)

    # Series 생성
    s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    print(f"Series:\n{s}\n")

    # 딕셔너리로 DataFrame 생성
    data = {
        '이름': ['김철수', '이영희', '박민수', '정수진', '최동욱'],
        '나이': [25, 30, 35, 28, 32],
        '도시': ['서울', '부산', '대구', '서울', '인천'],
        '점수': [85, 92, 78, 95, 88]
    }
    df = pd.DataFrame(data)
    print(f"DataFrame:\n{df}\n")

    # 리스트로 생성
    df2 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=['A', 'B', 'C'],
        index=['row1', 'row2', 'row3']
    )
    print(f"리스트로 생성:\n{df2}")

    return df


# =============================================================================
# 2. 데이터 확인
# =============================================================================
def inspect_data(df):
    """데이터 확인 메서드"""
    print("\n[2] 데이터 확인")
    print("=" * 50)

    print(f"처음 2행:\n{df.head(2)}\n")
    print(f"마지막 2행:\n{df.tail(2)}\n")
    print(f"형태: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    print(f"인덱스: {df.index.tolist()}")
    print(f"\n데이터 타입:\n{df.dtypes}\n")
    print(f"기본 통계:\n{df.describe()}\n")
    print(f"정보:")
    df.info()


# =============================================================================
# 3. 인덱싱과 선택
# =============================================================================
def indexing_selection(df):
    """인덱싱과 선택"""
    print("\n[3] 인덱싱과 선택")
    print("=" * 50)

    print(f"원본 DataFrame:\n{df}\n")

    # 컬럼 선택
    print(f"df['이름']:\n{df['이름']}\n")
    print(f"df[['이름', '나이']]:\n{df[['이름', '나이']]}\n")

    # 행 선택 (loc: 라벨 기반, iloc: 위치 기반)
    print(f"df.loc[0]:\n{df.loc[0]}\n")  # 첫 번째 행
    print(f"df.iloc[0:2]:\n{df.iloc[0:2]}\n")  # 처음 2행
    print(f"df.loc[0, '이름'] = {df.loc[0, '이름']}")  # 특정 값
    print(f"df.iloc[0, 1] = {df.iloc[0, 1]}")  # 위치로 접근

    # 조건부 선택
    print(f"\ndf[df['나이'] > 28]:\n{df[df['나이'] > 28]}")
    print(f"\ndf[(df['나이'] > 25) & (df['도시'] == '서울')]:\n{df[(df['나이'] > 25) & (df['도시'] == '서울')]}")

    # 쿼리 메서드
    result = df.query("나이 > 28 and 점수 >= 90")
    print(f"\ndf.query(\"나이 > 28 and 점수 >= 90\"):\n{result}")


# =============================================================================
# 4. 데이터 수정
# =============================================================================
def modify_data():
    """데이터 수정"""
    print("\n[4] 데이터 수정")
    print("=" * 50)

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    print(f"원본:\n{df}\n")

    # 컬럼 추가
    df['D'] = df['A'] + df['B']
    print(f"컬럼 추가 (D = A + B):\n{df}\n")

    # 컬럼 삭제
    df_dropped = df.drop('D', axis=1)
    print(f"컬럼 삭제:\n{df_dropped}\n")

    # 값 변경
    df.loc[0, 'A'] = 100
    print(f"값 변경 (df.loc[0, 'A'] = 100):\n{df}\n")

    # 조건부 변경
    df.loc[df['B'] > 4, 'C'] = 0
    print(f"조건부 변경:\n{df}\n")

    # 컬럼 이름 변경
    df_renamed = df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
    print(f"컬럼 이름 변경:\n{df_renamed}")


# =============================================================================
# 5. 결측치 처리
# =============================================================================
def handle_missing():
    """결측치 처리"""
    print("\n[5] 결측치 처리")
    print("=" * 50)

    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    })
    print(f"원본 (NaN 포함):\n{df}\n")

    # 결측치 확인
    print(f"결측치 확인:\n{df.isnull()}\n")
    print(f"컬럼별 결측치 수:\n{df.isnull().sum()}\n")

    # 결측치 제거
    df_dropna = df.dropna()
    print(f"dropna() - 행 제거:\n{df_dropna}\n")

    # 결측치 채우기
    df_filled = df.fillna(0)
    print(f"fillna(0):\n{df_filled}\n")

    df_ffill = df.fillna(method='ffill')
    print(f"fillna(method='ffill') - 앞의 값으로:\n{df_ffill}\n")

    df_mean = df.fillna(df.mean())
    print(f"fillna(df.mean()) - 평균으로:\n{df_mean}")


# =============================================================================
# 6. 그룹화와 집계
# =============================================================================
def groupby_aggregation():
    """그룹화와 집계"""
    print("\n[6] 그룹화와 집계")
    print("=" * 50)

    df = pd.DataFrame({
        '부서': ['영업', '개발', '영업', '개발', '영업', '개발'],
        '이름': ['김철수', '이영희', '박민수', '정수진', '최동욱', '강미영'],
        '매출': [100, 80, 120, 90, 110, 85],
        '경력': [3, 5, 7, 4, 6, 2]
    })
    print(f"원본:\n{df}\n")

    # 기본 그룹화
    grouped = df.groupby('부서')
    print(f"부서별 매출 합계:\n{grouped['매출'].sum()}\n")
    print(f"부서별 매출 평균:\n{grouped['매출'].mean()}\n")

    # 여러 집계 함수
    agg_result = grouped.agg({
        '매출': ['sum', 'mean', 'max'],
        '경력': ['mean', 'min', 'max']
    })
    print(f"다중 집계:\n{agg_result}\n")

    # 여러 컬럼으로 그룹화
    df['연도'] = [2023, 2023, 2024, 2024, 2023, 2024]
    multi_group = df.groupby(['부서', '연도'])['매출'].sum()
    print(f"부서, 연도별 매출:\n{multi_group}")


# =============================================================================
# 7. 정렬과 순위
# =============================================================================
def sorting_ranking():
    """정렬과 순위"""
    print("\n[7] 정렬과 순위")
    print("=" * 50)

    df = pd.DataFrame({
        '이름': ['A', 'B', 'C', 'D', 'E'],
        '점수': [85, 92, 78, 95, 88],
        '나이': [25, 30, 25, 35, 28]
    })
    print(f"원본:\n{df}\n")

    # 단일 컬럼 정렬
    sorted_df = df.sort_values('점수', ascending=False)
    print(f"점수 내림차순:\n{sorted_df}\n")

    # 여러 컬럼 정렬
    sorted_df2 = df.sort_values(['나이', '점수'], ascending=[True, False])
    print(f"나이 오름차순, 점수 내림차순:\n{sorted_df2}\n")

    # 인덱스 정렬
    df_shuffled = df.sample(frac=1)
    print(f"셔플된 데이터:\n{df_shuffled}")
    print(f"인덱스 정렬:\n{df_shuffled.sort_index()}\n")

    # 순위
    df['순위'] = df['점수'].rank(ascending=False)
    print(f"순위 추가:\n{df}")


# =============================================================================
# 8. 데이터 병합
# =============================================================================
def merge_data():
    """데이터 병합"""
    print("\n[8] 데이터 병합")
    print("=" * 50)

    # 두 DataFrame 준비
    df1 = pd.DataFrame({
        '사원ID': [1, 2, 3, 4],
        '이름': ['김철수', '이영희', '박민수', '정수진']
    })

    df2 = pd.DataFrame({
        '사원ID': [2, 3, 4, 5],
        '부서': ['개발', '영업', '마케팅', 'HR']
    })

    print(f"df1:\n{df1}\n")
    print(f"df2:\n{df2}\n")

    # Inner Join
    inner = pd.merge(df1, df2, on='사원ID', how='inner')
    print(f"Inner Join:\n{inner}\n")

    # Left Join
    left = pd.merge(df1, df2, on='사원ID', how='left')
    print(f"Left Join:\n{left}\n")

    # Outer Join
    outer = pd.merge(df1, df2, on='사원ID', how='outer')
    print(f"Outer Join:\n{outer}\n")

    # Concat
    df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

    concat_rows = pd.concat([df_a, df_b], ignore_index=True)
    print(f"세로 연결 (concat):\n{concat_rows}\n")

    concat_cols = pd.concat([df_a, df_b], axis=1)
    print(f"가로 연결 (concat, axis=1):\n{concat_cols}")


# =============================================================================
# 9. 피벗 테이블
# =============================================================================
def pivot_tables():
    """피벗 테이블"""
    print("\n[9] 피벗 테이블")
    print("=" * 50)

    df = pd.DataFrame({
        '날짜': ['2024-01', '2024-01', '2024-02', '2024-02'] * 2,
        '지역': ['서울', '부산'] * 4,
        '제품': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        '매출': [100, 80, 120, 90, 60, 70, 80, 50]
    })
    print(f"원본:\n{df}\n")

    # 피벗 테이블
    pivot = df.pivot_table(
        values='매출',
        index='지역',
        columns='제품',
        aggfunc='sum'
    )
    print(f"피벗 테이블 (지역 x 제품):\n{pivot}\n")

    # 복합 피벗
    pivot2 = df.pivot_table(
        values='매출',
        index=['날짜', '지역'],
        columns='제품',
        aggfunc=['sum', 'mean']
    )
    print(f"복합 피벗 테이블:\n{pivot2}")


# =============================================================================
# 10. 시계열 데이터
# =============================================================================
def time_series():
    """시계열 데이터"""
    print("\n[10] 시계열 데이터")
    print("=" * 50)

    # 날짜 범위 생성
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    print(f"날짜 범위:\n{dates}\n")

    # 시계열 DataFrame
    df = pd.DataFrame({
        '날짜': dates,
        '값': np.random.randn(10).cumsum()
    })
    df['날짜'] = pd.to_datetime(df['날짜'])
    df.set_index('날짜', inplace=True)
    print(f"시계열 데이터:\n{df}\n")

    # 리샘플링
    df_monthly = pd.DataFrame({
        '값': np.random.randn(100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='D'))

    monthly_mean = df_monthly.resample('M').mean()
    print(f"월별 평균:\n{monthly_mean}\n")

    # 이동 평균
    df['이동평균'] = df['값'].rolling(window=3).mean()
    print(f"이동 평균 (window=3):\n{df}")


# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("Pandas 기초 예제")
    print("=" * 60)

    df = create_dataframe()
    inspect_data(df)
    indexing_selection(df)
    modify_data()
    handle_missing()
    groupby_aggregation()
    sorting_ranking()
    merge_data()
    pivot_tables()
    time_series()

    print("\n" + "=" * 60)
    print("Pandas 핵심 정리")
    print("=" * 60)
    print("""
    핵심 자료구조:
    - Series: 1차원 (라벨이 붙은 배열)
    - DataFrame: 2차원 (표 형식)

    자주 사용하는 메서드:
    - 확인: head, tail, info, describe, shape
    - 선택: loc (라벨), iloc (위치), query
    - 수정: drop, rename, fillna
    - 집계: groupby, agg, pivot_table
    - 병합: merge, concat, join

    팁:
    - 체이닝: df.dropna().groupby('col').mean()
    - 복사: df.copy() vs 뷰 (슬라이싱)
    - 메모리: category 타입으로 문자열 절약
    - 성능: apply 대신 벡터화 연산 사용
    """)


if __name__ == "__main__":
    main()
