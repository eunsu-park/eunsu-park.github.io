"""
18. 카메라 캘리브레이션
- 카메라 내부 파라미터
- 왜곡 보정
- 체스보드 검출
- 스테레오 비전 기초
"""

import cv2
import numpy as np


def camera_model_concept():
    """카메라 모델 개념"""
    print("=" * 50)
    print("카메라 모델 개념")
    print("=" * 50)

    print("\n1. 핀홀 카메라 모델")
    print("   - 3D 점 → 2D 이미지 투영")
    print("   - 원근 투영 (Perspective Projection)")

    print("\n2. 카메라 내부 파라미터 (Intrinsic)")
    print("""
   K = | fx  0  cx |
       |  0 fy  cy |
       |  0  0   1 |

   fx, fy: 초점 거리 (픽셀 단위)
   cx, cy: 주점 (principal point, 이미지 중심)
""")

    print("3. 카메라 외부 파라미터 (Extrinsic)")
    print("   - R: 회전 행렬 (3x3)")
    print("   - t: 평행 이동 벡터 (3x1)")
    print("   - 월드 좌표 → 카메라 좌표 변환")

    print("\n4. 투영 행렬")
    print("   P = K[R|t]")
    print("   p = P * X  (동차 좌표계)")

    print("\n5. 왜곡 계수 (Distortion Coefficients)")
    print("   - k1, k2, k3: 방사 왜곡 (radial)")
    print("   - p1, p2: 접선 왜곡 (tangential)")
    print("   - dist_coeffs = [k1, k2, p1, p2, k3]")


def create_chessboard_image():
    """체스보드 이미지 생성"""
    # 체스보드 패턴
    rows, cols = 7, 9
    square_size = 40

    img = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)
    img[:] = [255, 255, 255]

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                x1, y1 = j * square_size, i * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return img


def chessboard_detection_demo():
    """체스보드 코너 검출 데모"""
    print("\n" + "=" * 50)
    print("체스보드 코너 검출")
    print("=" * 50)

    # 체스보드 이미지 생성
    chessboard = create_chessboard_image()

    # 약간의 원근 변환 적용 (실제 촬영 시뮬레이션)
    h, w = chessboard.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[20, 30], [w-30, 20], [w-20, h-10], [10, h-30]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(chessboard, M, (w, h), borderValue=(200, 200, 200))

    # 그레이스케일 변환
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 코너 검출 파라미터
    # 내부 코너 수 (검은색-흰색 교차점)
    pattern_size = (8, 6)  # 가로 8, 세로 6 코너

    # 코너 검출
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"패턴 크기: {pattern_size}")
    print(f"코너 검출 성공: {found}")

    if found:
        # 서브픽셀 정밀도로 개선
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        print(f"검출된 코너 수: {len(corners)}")

        # 결과 시각화
        result = warped.copy()
        cv2.drawChessboardCorners(result, pattern_size, corners, found)

        cv2.imwrite('chessboard_input.jpg', warped)
        cv2.imwrite('chessboard_corners.jpg', result)
        print("이미지 저장 완료")

    print("\n검출 플래그:")
    print("  CALIB_CB_ADAPTIVE_THRESH: 적응적 이진화")
    print("  CALIB_CB_NORMALIZE_IMAGE: 이미지 정규화")
    print("  CALIB_CB_FAST_CHECK: 빠른 체크 (실패 시 조기 종료)")


def camera_calibration_simulation():
    """카메라 캘리브레이션 시뮬레이션"""
    print("\n" + "=" * 50)
    print("카메라 캘리브레이션 시뮬레이션")
    print("=" * 50)

    # 체스보드 파라미터
    pattern_size = (8, 6)
    square_size = 1.0  # 실제 사각형 크기 (단위: cm, mm 등)

    # 3D 객체 점 (체스보드 평면, z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    print(f"객체 점 (3D): {objp.shape}")
    print(f"첫 번째 점: {objp[0]}")
    print(f"마지막 점: {objp[-1]}")

    # 시뮬레이션용 여러 이미지에서 검출된 점들
    objpoints = []  # 3D 점들
    imgpoints = []  # 2D 점들

    # 시뮬레이션 (실제로는 여러 각도에서 촬영한 이미지 사용)
    chessboard = create_chessboard_image()
    gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"\n사용된 이미지 수: {len(objpoints)}")

        # 캘리브레이션 (최소 3-5개 이미지 필요)
        # 여기서는 시뮬레이션이므로 실제 결과와 다를 수 있음
        h, w = gray.shape

        # 초기 카메라 행렬 추정
        fx = fy = w  # 대략적 초점 거리
        cx, cy = w/2, h/2

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(5)

        print("\n추정된 카메라 행렬:")
        print(camera_matrix)

        print("\n캘리브레이션 프로세스:")
        print("  1. 여러 각도에서 체스보드 촬영 (10-20장)")
        print("  2. 각 이미지에서 코너 검출")
        print("  3. cv2.calibrateCamera() 호출")
        print("  4. 카메라 행렬, 왜곡 계수 획득")


def calibration_workflow():
    """캘리브레이션 워크플로우"""
    print("\n" + "=" * 50)
    print("실제 캘리브레이션 워크플로우")
    print("=" * 50)

    code = '''
import cv2
import numpy as np
import glob

# 체스보드 설정
pattern_size = (9, 6)  # 내부 코너 수
square_size = 25.0     # mm 단위

# 객체 점 생성
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D
imgpoints = []  # 2D

# 이미지 로드 및 코너 검출
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

# 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"RMS error: {ret}")
print(f"Camera matrix:\\n{camera_matrix}")
print(f"Distortion coefficients:\\n{dist_coeffs}")

# 결과 저장
np.savez('calibration.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)
'''

    print(code)

    print("\n캘리브레이션 팁:")
    print("  1. 최소 10-20장의 이미지 사용")
    print("  2. 다양한 각도와 위치에서 촬영")
    print("  3. 체스보드가 이미지 전체에 분포하도록")
    print("  4. 조명이 균일해야 함")
    print("  5. 블러 없이 선명한 이미지")


def undistort_demo():
    """왜곡 보정 데모"""
    print("\n" + "=" * 50)
    print("왜곡 보정 (Undistortion)")
    print("=" * 50)

    # 시뮬레이션용 왜곡 이미지 생성
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 격자 패턴
    for i in range(0, 600, 50):
        cv2.line(img, (i, 0), (i, 400), (0, 0, 0), 1)
    for j in range(0, 400, 50):
        cv2.line(img, (0, j), (600, j), (0, 0, 0), 1)

    # 가상의 왜곡 적용 (barrel distortion 시뮬레이션)
    h, w = img.shape[:2]
    camera_matrix = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], dtype=np.float64)

    # 왜곡 계수 (k1이 음수면 barrel, 양수면 pincushion)
    dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0])

    # 왜곡 적용 (역으로 undistort 사용)
    distorted = cv2.undistort(img, camera_matrix, -dist_coeffs)

    # 왜곡 보정
    undistorted = cv2.undistort(distorted, camera_matrix, dist_coeffs)

    cv2.imwrite('undistort_original.jpg', img)
    cv2.imwrite('undistort_distorted.jpg', distorted)
    cv2.imwrite('undistort_corrected.jpg', undistorted)

    print("왜곡 보정 방법:")
    print("  1. cv2.undistort()")
    print("     - 간단하게 사용")
    print("     - 매번 계산")

    print("\n  2. cv2.initUndistortRectifyMap() + cv2.remap()")
    print("     - 맵을 미리 계산")
    print("     - 비디오에서 효율적")

    code = '''
# 효율적인 방법 (비디오용)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None,
    camera_matrix, (w, h), cv2.CV_32FC1
)

# 프레임마다 적용
undistorted = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR)
'''
    print(code)


def stereo_vision_concept():
    """스테레오 비전 개념"""
    print("\n" + "=" * 50)
    print("스테레오 비전 기초")
    print("=" * 50)

    print("\n1. 스테레오 비전 원리")
    print("   - 두 카메라로 동일 장면 촬영")
    print("   - 시차(disparity)로 깊이 계산")
    print("   - depth = (baseline * focal_length) / disparity")

    print("\n2. 스테레오 캘리브레이션")
    print("   - 각 카메라 개별 캘리브레이션")
    print("   - 스테레오 쌍 캘리브레이션")
    print("   - 에피폴라 기하학 계산")

    code = '''
# 스테레오 캘리브레이션
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, D1, K2, D2, image_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)
'''
    print(code)

    print("\n3. 스테레오 정합 (Stereo Rectification)")
    print("   - 두 이미지를 같은 평면으로 정렬")
    print("   - 수평선상에서만 매칭 탐색")

    print("\n4. 시차 맵 계산")
    print("   - StereoBM: Block Matching (빠름)")
    print("   - StereoSGBM: Semi-Global BM (정확)")


def stereo_matching_demo():
    """스테레오 매칭 시뮬레이션"""
    print("\n" + "=" * 50)
    print("스테레오 매칭 시뮬레이션")
    print("=" * 50)

    # 시뮬레이션용 스테레오 이미지 쌍 생성
    left = np.zeros((300, 400), dtype=np.uint8)
    left[:] = 150
    cv2.rectangle(left, (100, 100), (200, 200), 80, -1)  # 가까운 객체
    cv2.rectangle(left, (250, 120), (350, 180), 100, -1)  # 먼 객체

    # 오른쪽 이미지 (시차 적용)
    right = np.zeros((300, 400), dtype=np.uint8)
    right[:] = 150
    cv2.rectangle(right, (80, 100), (180, 200), 80, -1)   # 시차 20 (가까움)
    cv2.rectangle(right, (240, 120), (340, 180), 100, -1)  # 시차 10 (멀음)

    # StereoBM
    stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity_bm = stereo_bm.compute(left, right)

    # StereoSGBM
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity_sgbm = stereo_sgbm.compute(left, right)

    # 정규화
    disparity_bm_norm = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_sgbm_norm = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite('stereo_left.jpg', left)
    cv2.imwrite('stereo_right.jpg', right)
    cv2.imwrite('stereo_disparity_bm.jpg', disparity_bm_norm)
    cv2.imwrite('stereo_disparity_sgbm.jpg', disparity_sgbm_norm)

    print("시차 맵 생성 완료")
    print("\nStereoBM 파라미터:")
    print("  numDisparities: 시차 범위 (16의 배수)")
    print("  blockSize: 매칭 블록 크기 (홀수, 5~21)")

    print("\nStereoSGBM 파라미터:")
    print("  P1, P2: 부드러움 제어")
    print("  uniquenessRatio: 매칭 고유성")
    print("  speckleWindowSize: 스페클 필터 크기")


def pose_estimation_concept():
    """포즈 추정 개념"""
    print("\n" + "=" * 50)
    print("포즈 추정 (Pose Estimation)")
    print("=" * 50)

    print("\n카메라 포즈 추정:")
    print("  - 3D-2D 대응점으로 카메라 위치/방향 추정")
    print("  - cv2.solvePnP()")

    code = '''
# 3D 객체 점 (알려진 월드 좌표)
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
], dtype=np.float32)

# 2D 이미지 점 (검출된 좌표)
image_points = np.array([...], dtype=np.float32)

# 포즈 추정
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points,
    camera_matrix, dist_coeffs
)

# 회전 벡터 → 회전 행렬
rotation_matrix, _ = cv2.Rodrigues(rvec)

# 3D 축 그리기
axis_points = np.float32([
    [3, 0, 0], [0, 3, 0], [0, 0, -3]
]).reshape(-1, 3)
imgpts, _ = cv2.projectPoints(
    axis_points, rvec, tvec, camera_matrix, dist_coeffs
)
'''
    print(code)

    print("\n활용:")
    print("  - AR (Augmented Reality)")
    print("  - 로봇 비전")
    print("  - 3D 재구성")


def main():
    """메인 함수"""
    # 카메라 모델 개념
    camera_model_concept()

    # 체스보드 검출
    chessboard_detection_demo()

    # 캘리브레이션 시뮬레이션
    camera_calibration_simulation()

    # 캘리브레이션 워크플로우
    calibration_workflow()

    # 왜곡 보정
    undistort_demo()

    # 스테레오 비전
    stereo_vision_concept()

    # 스테레오 매칭
    stereo_matching_demo()

    # 포즈 추정
    pose_estimation_concept()

    print("\n카메라 캘리브레이션 데모 완료!")


if __name__ == '__main__':
    main()
