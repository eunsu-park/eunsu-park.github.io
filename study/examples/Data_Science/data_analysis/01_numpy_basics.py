"""
NumPy 기초 (NumPy Basics)
Fundamental NumPy Operations

NumPy는 Python의 핵심 과학 컴퓨팅 라이브러리입니다.
"""

import numpy as np


# =============================================================================
# 1. 배열 생성
# =============================================================================
def array_creation():
    """다양한 배열 생성 방법"""
    print("\n[1] 배열 생성")
    print("=" * 50)

    # 리스트로부터 생성
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"리스트로부터: {arr1}")

    # 2D 배열
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D 배열:\n{arr2d}")

    # 특수 배열
    zeros = np.zeros((3, 4))  # 0으로 채움
    ones = np.ones((2, 3))    # 1로 채움
    empty = np.empty((2, 2))  # 초기화되지 않은 배열
    full = np.full((2, 3), 7) # 특정 값으로 채움
    eye = np.eye(3)           # 단위 행렬

    print(f"\nnp.zeros((3,4)):\n{zeros}")
    print(f"\nnp.eye(3):\n{eye}")

    # 수열 생성
    arange = np.arange(0, 10, 2)  # start, stop, step
    linspace = np.linspace(0, 1, 5)  # start, stop, num
    logspace = np.logspace(0, 3, 4)  # 10^0 ~ 10^3

    print(f"\nnp.arange(0, 10, 2): {arange}")
    print(f"np.linspace(0, 1, 5): {linspace}")
    print(f"np.logspace(0, 3, 4): {logspace}")

    # 난수 배열
    rand = np.random.rand(3, 3)        # 균일 분포 [0, 1)
    randn = np.random.randn(3, 3)      # 표준 정규 분포
    randint = np.random.randint(0, 10, (3, 3))  # 정수

    print(f"\nnp.random.rand(3,3):\n{rand}")


# =============================================================================
# 2. 배열 속성과 형태 변환
# =============================================================================
def array_attributes():
    """배열 속성 및 형태 변환"""
    print("\n[2] 배열 속성과 형태 변환")
    print("=" * 50)

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print(f"배열:\n{arr}")
    print(f"shape: {arr.shape}")      # (2, 3)
    print(f"ndim: {arr.ndim}")        # 차원 수
    print(f"size: {arr.size}")        # 전체 요소 수
    print(f"dtype: {arr.dtype}")      # 데이터 타입

    # 형태 변환
    reshaped = arr.reshape(3, 2)
    print(f"\nreshape(3, 2):\n{reshaped}")

    flattened = arr.flatten()
    print(f"flatten(): {flattened}")

    raveled = arr.ravel()
    print(f"ravel(): {raveled}")

    # 전치
    transposed = arr.T
    print(f"\n전치 (T):\n{transposed}")

    # 차원 추가/제거
    arr1d = np.array([1, 2, 3])
    expanded = np.expand_dims(arr1d, axis=0)  # (3,) -> (1, 3)
    print(f"\nexpand_dims: {arr1d.shape} -> {expanded.shape}")

    squeezed = np.squeeze(expanded)
    print(f"squeeze: {expanded.shape} -> {squeezed.shape}")


# =============================================================================
# 3. 인덱싱과 슬라이싱
# =============================================================================
def indexing_slicing():
    """인덱싱과 슬라이싱"""
    print("\n[3] 인덱싱과 슬라이싱")
    print("=" * 50)

    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

    print(f"원본 배열:\n{arr}")

    # 기본 인덱싱
    print(f"\narr[0, 0] = {arr[0, 0]}")  # 첫 번째 요소
    print(f"arr[1, 2] = {arr[1, 2]}")    # 7
    print(f"arr[-1, -1] = {arr[-1, -1]}")  # 마지막 요소

    # 슬라이싱
    print(f"\narr[0, :] = {arr[0, :]}")    # 첫 번째 행
    print(f"arr[:, 0] = {arr[:, 0]}")      # 첫 번째 열
    print(f"arr[0:2, 1:3] =\n{arr[0:2, 1:3]}")  # 부분 배열

    # 팬시 인덱싱
    indices = [0, 2]
    print(f"\narr[indices] =\n{arr[indices]}")  # 0, 2번째 행

    # 불리안 인덱싱
    mask = arr > 5
    print(f"\nmask (arr > 5):\n{mask}")
    print(f"arr[arr > 5] = {arr[arr > 5]}")


# =============================================================================
# 4. 배열 연산
# =============================================================================
def array_operations():
    """배열 연산과 브로드캐스팅"""
    print("\n[4] 배열 연산")
    print("=" * 50)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print(f"a =\n{a}")
    print(f"b =\n{b}")

    # 요소별 연산
    print(f"\na + b =\n{a + b}")
    print(f"a * b =\n{a * b}")    # 요소별 곱셈
    print(f"a / b =\n{a / b}")
    print(f"a ** 2 =\n{a ** 2}")

    # 행렬 연산
    print(f"\na @ b (행렬 곱) =\n{a @ b}")
    print(f"np.dot(a, b) =\n{np.dot(a, b)}")

    # 브로드캐스팅
    print("\n[브로드캐스팅]")
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 10
    row_vec = np.array([1, 0, 1])
    col_vec = np.array([[10], [20]])

    print(f"arr + scalar =\n{arr + scalar}")
    print(f"arr * row_vec =\n{arr * row_vec}")
    print(f"arr + col_vec =\n{arr + col_vec}")


# =============================================================================
# 5. 수학 함수
# =============================================================================
def math_functions():
    """수학 함수"""
    print("\n[5] 수학 함수")
    print("=" * 50)

    arr = np.array([1, 4, 9, 16, 25])
    print(f"arr = {arr}")

    # 기본 수학 함수
    print(f"\nnp.sqrt(arr) = {np.sqrt(arr)}")
    print(f"np.exp(arr[:3]) = {np.exp(arr[:3])}")
    print(f"np.log(arr) = {np.log(arr)}")

    # 삼각 함수
    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"\nangles = {np.degrees(angles)}")
    print(f"np.sin(angles) = {np.sin(angles)}")

    # 반올림
    float_arr = np.array([1.2, 2.5, 3.7, -1.2])
    print(f"\nfloat_arr = {float_arr}")
    print(f"np.round(float_arr) = {np.round(float_arr)}")
    print(f"np.floor(float_arr) = {np.floor(float_arr)}")
    print(f"np.ceil(float_arr) = {np.ceil(float_arr)}")


# =============================================================================
# 6. 집계 함수
# =============================================================================
def aggregation_functions():
    """집계 함수"""
    print("\n[6] 집계 함수")
    print("=" * 50)

    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    print(f"배열:\n{arr}")

    # 전체 집계
    print(f"\nnp.sum(arr) = {np.sum(arr)}")
    print(f"np.mean(arr) = {np.mean(arr)}")
    print(f"np.std(arr) = {np.std(arr):.4f}")
    print(f"np.min(arr) = {np.min(arr)}")
    print(f"np.max(arr) = {np.max(arr)}")

    # 축별 집계
    print(f"\nnp.sum(arr, axis=0) = {np.sum(arr, axis=0)}")  # 열 합
    print(f"np.sum(arr, axis=1) = {np.sum(arr, axis=1)}")    # 행 합
    print(f"np.mean(arr, axis=0) = {np.mean(arr, axis=0)}")

    # 누적
    print(f"\nnp.cumsum(arr.flatten()) = {np.cumsum(arr.flatten())}")

    # 위치 찾기
    print(f"\nnp.argmax(arr) = {np.argmax(arr)}")  # 평탄화된 인덱스
    print(f"np.argmin(arr) = {np.argmin(arr)}")


# =============================================================================
# 7. 배열 조작
# =============================================================================
def array_manipulation():
    """배열 조작"""
    print("\n[7] 배열 조작")
    print("=" * 50)

    # 연결
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print(f"a =\n{a}")
    print(f"b =\n{b}")

    vstack = np.vstack([a, b])  # 수직 연결
    hstack = np.hstack([a, b])  # 수평 연결
    concat = np.concatenate([a, b], axis=0)

    print(f"\nvstack:\n{vstack}")
    print(f"\nhstack:\n{hstack}")

    # 분할
    arr = np.arange(16).reshape(4, 4)
    print(f"\n분할 대상:\n{arr}")

    vsplit = np.vsplit(arr, 2)
    hsplit = np.hsplit(arr, 2)

    print(f"\nvsplit(2):\n{vsplit[0]}\n{vsplit[1]}")

    # 정렬
    unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"\n정렬 전: {unsorted}")
    print(f"np.sort(): {np.sort(unsorted)}")
    print(f"np.argsort(): {np.argsort(unsorted)}")  # 정렬된 인덱스


# =============================================================================
# 8. 선형 대수
# =============================================================================
def linear_algebra():
    """선형 대수 연산"""
    print("\n[8] 선형 대수")
    print("=" * 50)

    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])

    print(f"A =\n{A}")
    print(f"b = {b}")

    # 행렬식
    det = np.linalg.det(A)
    print(f"\n행렬식 det(A) = {det:.4f}")

    # 역행렬
    A_inv = np.linalg.inv(A)
    print(f"\n역행렬 A^(-1) =\n{A_inv}")

    # 선형 시스템 풀기: Ax = b
    x = np.linalg.solve(A, b)
    print(f"\nAx = b의 해 x = {x}")
    print(f"검증 A @ x = {A @ x}")

    # 고유값/고유벡터
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\n고유값: {eigenvalues}")
    print(f"고유벡터:\n{eigenvectors}")

    # 특이값 분해 (SVD)
    U, S, Vt = np.linalg.svd(A)
    print(f"\nSVD:")
    print(f"U =\n{U}")
    print(f"S = {S}")
    print(f"Vt =\n{Vt}")


# =============================================================================
# 9. 실전 예제
# =============================================================================
def practical_examples():
    """실전 예제"""
    print("\n[9] 실전 예제")
    print("=" * 50)

    # 예제 1: 유클리드 거리
    print("\n예제 1: 유클리드 거리")
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 6])
    distance = np.linalg.norm(point1 - point2)
    print(f"점1: {point1}, 점2: {point2}")
    print(f"거리: {distance:.4f}")

    # 예제 2: 이동 평균
    print("\n예제 2: 이동 평균")
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window = 3
    moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
    print(f"데이터: {data}")
    print(f"이동 평균 (window={window}): {moving_avg}")

    # 예제 3: 정규화
    print("\n예제 3: 정규화")
    data = np.array([10, 20, 30, 40, 50])
    normalized = (data - data.mean()) / data.std()
    min_max = (data - data.min()) / (data.max() - data.min())
    print(f"원본: {data}")
    print(f"Z-score: {normalized}")
    print(f"Min-Max: {min_max}")

    # 예제 4: 행렬 방정식
    print("\n예제 4: 연립방정식 풀기")
    # 2x + 3y = 8
    # 3x + 4y = 11
    A = np.array([[2, 3], [3, 4]])
    b = np.array([8, 11])
    solution = np.linalg.solve(A, b)
    print(f"2x + 3y = 8")
    print(f"3x + 4y = 11")
    print(f"해: x = {solution[0]:.4f}, y = {solution[1]:.4f}")


# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("NumPy 기초 예제")
    print("=" * 60)

    array_creation()
    array_attributes()
    indexing_slicing()
    array_operations()
    math_functions()
    aggregation_functions()
    array_manipulation()
    linear_algebra()
    practical_examples()

    print("\n" + "=" * 60)
    print("NumPy 핵심 정리")
    print("=" * 60)
    print("""
    핵심 개념:
    - ndarray: N차원 배열 (효율적인 메모리 사용)
    - 브로드캐스팅: 다른 형태의 배열 간 연산
    - 벡터화: 루프 없이 빠른 연산

    자주 사용하는 함수:
    - 생성: array, zeros, ones, arange, linspace
    - 형태: reshape, flatten, T
    - 집계: sum, mean, std, min, max
    - 연산: +, -, *, /, @, dot
    - 선형대수: linalg.inv, linalg.solve, linalg.eig

    팁:
    - 루프 대신 벡터화 연산 사용 (훨씬 빠름)
    - axis 매개변수 이해하기 (0=행 방향, 1=열 방향)
    - 복사 vs 뷰 구분하기 (copy() vs 슬라이싱)
    """)


if __name__ == "__main__":
    main()
