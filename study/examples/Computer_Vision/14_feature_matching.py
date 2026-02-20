"""
14. 특징점 매칭
- BFMatcher (Brute Force)
- FLANN 매처
- 좋은 매칭 선별 (ratio test)
- Homography 계산
"""

import cv2
import numpy as np


def create_test_images():
    """매칭용 테스트 이미지 쌍 생성"""
    # 원본 이미지
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img1[:] = [200, 200, 200]

    # 특징이 있는 패턴
    cv2.rectangle(img1, (50, 50), (150, 150), (50, 50, 50), -1)
    cv2.circle(img1, (250, 100), 40, (100, 100, 100), -1)
    cv2.rectangle(img1, (300, 150), (380, 250), (80, 80, 80), -1)

    # 체커보드 패턴 추가
    for i in range(3):
        for j in range(3):
            x, y = 100 + i * 30, 180 + j * 30
            if (i + j) % 2 == 0:
                cv2.rectangle(img1, (x, y), (x + 30, y + 30), (0, 0, 0), -1)

    cv2.putText(img1, 'MATCH', (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 변환된 이미지 (회전 + 스케일)
    h, w = img1.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 0.9)  # 15도 회전, 0.9배 스케일
    img2 = cv2.warpAffine(img1, M, (w, h), borderValue=(200, 200, 200))

    return img1, img2


def bf_matcher_demo():
    """Brute Force 매처 데모"""
    print("=" * 50)
    print("BFMatcher (Brute Force Matcher)")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB 특징점 검출
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # BFMatcher 생성
    # NORM_HAMMING: 이진 디스크립터용 (ORB, BRIEF)
    # NORM_L2: 실수 디스크립터용 (SIFT, SURF)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 매칭
    matches = bf.match(des1, des2)

    # 거리순 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"키포인트: img1={len(kp1)}, img2={len(kp2)}")
    print(f"매칭 수: {len(matches)}")

    # 상위 매칭 정보
    print("\n상위 5개 매칭:")
    for i, m in enumerate(matches[:5]):
        print(f"  {i}: queryIdx={m.queryIdx}, trainIdx={m.trainIdx}, distance={m.distance:.1f}")

    # 결과 그리기
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nBFMatcher 특성:")
    print("  - 모든 쌍 비교 (O(n*m))")
    print("  - crossCheck=True: 상호 최근접만 선택")
    print("  - 정확하지만 느림")

    cv2.imwrite('bf_match_img1.jpg', img1)
    cv2.imwrite('bf_match_img2.jpg', img2)
    cv2.imwrite('bf_match_result.jpg', result)


def knn_match_demo():
    """KNN 매칭과 Ratio Test 데모"""
    print("\n" + "=" * 50)
    print("KNN 매칭 + Ratio Test")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # BFMatcher (crossCheck=False for knnMatch)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN 매칭 (k=2)
    matches = bf.knnMatch(des1, des2, k=2)

    print(f"KNN 매칭 수: {len(matches)}")

    # Lowe's Ratio Test
    # 최근접 거리 / 차선 거리 < threshold
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Ratio Test 후: {len(good_matches)}")

    # 결과 그리기
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nRatio Test (Lowe's ratio):")
    print("  - 최근접 / 차선 < 0.75 (보통 0.7~0.8)")
    print("  - 모호한 매칭 제거")
    print("  - 오매칭 감소")

    cv2.imwrite('knn_match_result.jpg', result)

    return kp1, kp2, good_matches


def flann_matcher_demo():
    """FLANN 매처 데모"""
    print("\n" + "=" * 50)
    print("FLANN Matcher")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        # SIFT 사용 (실수 디스크립터)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # FLANN 파라미터 (SIFT/SURF용)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

    except AttributeError:
        # SIFT 없으면 ORB 사용
        print("SIFT 없음, ORB 사용")
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # FLANN 파라미터 (ORB용)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)

    # FLANN 매처 생성
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN 매칭
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio Test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"FLANN 매칭: {len(matches)} → 좋은 매칭: {len(good_matches)}")

    # 결과 그리기
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nFLANN 특성:")
    print("  - 근사 최근접 이웃 탐색")
    print("  - 대규모 데이터에 효율적")
    print("  - KD-Tree (SIFT) 또는 LSH (ORB)")

    cv2.imwrite('flann_match_result.jpg', result)


def homography_demo():
    """호모그래피 계산 데모"""
    print("\n" + "=" * 50)
    print("호모그래피 (Homography)")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"좋은 매칭 수: {len(good_matches)}")

    if len(good_matches) >= 4:
        # 매칭점 좌표 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 호모그래피 계산 (RANSAC)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        inliers = sum(matches_mask)
        print(f"인라이어: {inliers}/{len(good_matches)}")

        if H is not None:
            print(f"\n호모그래피 행렬:\n{H}")

            # img1의 경계를 img2에 투영
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            # img2에 경계 그리기
            result = img2.copy()
            dst = np.int32(dst)
            cv2.polylines(result, [dst], True, (0, 255, 0), 3)

            cv2.imwrite('homography_result.jpg', result)

            # 매칭 시각화 (인라이어만)
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matches_mask,
                flags=2
            )
            match_result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            cv2.imwrite('homography_matches.jpg', match_result)

    print("\n호모그래피 용도:")
    print("  - 이미지 정합 (Image Registration)")
    print("  - 파노라마 스티칭")
    print("  - 객체 인식 (위치 추정)")
    print("  - 증강 현실")


def match_object_demo():
    """객체 매칭 데모"""
    print("\n" + "=" * 50)
    print("객체 매칭 실습")
    print("=" * 50)

    # 템플릿 이미지 (찾을 객체)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    template[:] = [200, 200, 200]
    cv2.rectangle(template, (10, 10), (90, 90), (50, 50, 50), -1)
    cv2.circle(template, (50, 50), 20, (100, 100, 100), -1)

    # 장면 이미지 (객체가 포함된)
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    scene[:] = [180, 180, 180]

    # 템플릿을 장면에 배치 (회전 및 스케일 적용)
    h, w = template.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    rotated_template = cv2.warpAffine(template, M, (w, h), borderValue=(180, 180, 180))

    # 장면에 붙이기
    scene[100:200, 150:250] = rotated_template

    # 다른 객체 추가 (방해물)
    cv2.circle(scene, (80, 80), 30, (120, 120, 120), -1)
    cv2.rectangle(scene, (300, 200), (380, 280), (90, 90, 90), -1)

    # 특징점 매칭
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_template, None)
    kp2, des2 = orb.detectAndCompute(gray_scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"템플릿 키포인트: {len(kp1)}")
    print(f"장면 키포인트: {len(kp2)}")
    print(f"좋은 매칭: {len(good)}")

    # 결과 시각화
    result = cv2.drawMatches(template, kp1, scene, kp2, good, None)
    cv2.imwrite('object_template.jpg', template)
    cv2.imwrite('object_scene.jpg', scene)
    cv2.imwrite('object_match.jpg', result)


def main():
    """메인 함수"""
    # BF 매처
    bf_matcher_demo()

    # KNN 매칭
    knn_match_demo()

    # FLANN 매처
    flann_matcher_demo()

    # 호모그래피
    homography_demo()

    # 객체 매칭
    match_object_demo()

    print("\n특징점 매칭 데모 완료!")


if __name__ == '__main__':
    main()
