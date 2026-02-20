"""
15. 객체 검출 기초
- Template Matching
- Haar Cascade
- HOG + SVM (개념)
"""

import cv2
import numpy as np


def create_scene_with_objects():
    """객체가 있는 장면 이미지 생성"""
    scene = np.zeros((400, 600, 3), dtype=np.uint8)
    scene[:] = [200, 200, 200]

    # 별 모양 객체 (찾을 대상)
    def draw_star(img, center, size, color):
        pts = []
        for i in range(5):
            outer = np.radians(i * 72 - 90)
            inner = np.radians(i * 72 + 36 - 90)
            pts.append([int(center[0] + size * np.cos(outer)),
                       int(center[1] + size * np.sin(outer))])
            pts.append([int(center[0] + size * 0.4 * np.cos(inner)),
                       int(center[1] + size * 0.4 * np.sin(inner))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)

    # 여러 별 배치
    draw_star(scene, (100, 100), 30, (0, 0, 150))
    draw_star(scene, (300, 200), 40, (0, 0, 180))
    draw_star(scene, (500, 300), 35, (0, 0, 160))

    # 방해 객체
    cv2.circle(scene, (200, 300), 40, (150, 0, 0), -1)
    cv2.rectangle(scene, (400, 50), (480, 130), (0, 150, 0), -1)

    # 템플릿 (별 하나)
    template = np.zeros((80, 80, 3), dtype=np.uint8)
    template[:] = [200, 200, 200]
    draw_star(template, (40, 40), 30, (0, 0, 150))

    return scene, template


def template_matching_demo():
    """템플릿 매칭 데모"""
    print("=" * 50)
    print("템플릿 매칭 (Template Matching)")
    print("=" * 50)

    scene, template = create_scene_with_objects()

    # 그레이스케일 변환
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape

    # 템플릿 매칭 방법들
    methods = [
        ('TM_CCOEFF', cv2.TM_CCOEFF),
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),
    ]

    print("\n매칭 방법 결과:")

    for name, method in methods:
        result = cv2.matchTemplate(scene_gray, template_gray, method)

        # 최소/최대 위치 찾기
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # SQDIFF는 최소값이 최적
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = min_val
        else:
            top_left = max_loc
            score = max_val

        print(f"  {name}: score={score:.4f}, loc={top_left}")

        # 결과 시각화
        scene_copy = scene.copy()
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(scene_copy, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(f'template_{name}.jpg', scene_copy)

    print("\n매칭 방법 특성:")
    print("  TM_SQDIFF: 차이 제곱합 (작을수록 좋음)")
    print("  TM_CCORR: 상관관계 (클수록 좋음)")
    print("  TM_CCOEFF: 상관계수 (클수록 좋음)")
    print("  _NORMED: 정규화 버전 (-1~1 또는 0~1)")

    cv2.imwrite('template_scene.jpg', scene)
    cv2.imwrite('template_template.jpg', template)


def multi_scale_template_demo():
    """다중 스케일 템플릿 매칭"""
    print("\n" + "=" * 50)
    print("다중 스케일 템플릿 매칭")
    print("=" * 50)

    scene, template = create_scene_with_objects()
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_val = -1
    best_scale = 1.0
    best_loc = (0, 0)

    # 다양한 스케일로 매칭
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    for scale in scales:
        # 템플릿 크기 조정
        new_w = int(template_gray.shape[1] * scale)
        new_h = int(template_gray.shape[0] * scale)

        if new_w > scene_gray.shape[1] or new_h > scene_gray.shape[0]:
            continue

        resized = cv2.resize(template_gray, (new_w, new_h))

        # 매칭
        result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        print(f"  Scale {scale:.2f}: score={max_val:.4f}")

        if max_val > best_val:
            best_val = max_val
            best_scale = scale
            best_loc = max_loc
            best_match = resized.shape

    print(f"\n최적 스케일: {best_scale}, score={best_val:.4f}")

    # 결과 그리기
    result_img = scene.copy()
    h, w = best_match
    cv2.rectangle(result_img, best_loc, (best_loc[0]+w, best_loc[1]+h), (0, 255, 0), 2)
    cv2.imwrite('multi_scale_result.jpg', result_img)


def find_all_matches_demo():
    """모든 매칭 찾기"""
    print("\n" + "=" * 50)
    print("모든 매칭 찾기")
    print("=" * 50)

    scene, template = create_scene_with_objects()
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape

    # 템플릿 매칭
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 임계값 이상인 모든 위치 찾기
    threshold = 0.6
    locations = np.where(result >= threshold)

    scene_copy = scene.copy()
    match_count = 0

    # 중복 제거를 위한 NMS 적용
    boxes = []
    for pt in zip(*locations[::-1]):
        boxes.append([pt[0], pt[1], pt[0]+w, pt[1]+h])

    # 간단한 NMS
    boxes = np.array(boxes)
    if len(boxes) > 0:
        # 점수순 정렬
        scores = [result[b[1], b[0]] for b in boxes]
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            # 다른 박스와의 겹침 계산
            remaining = []
            for j in indices[1:]:
                # IoU 계산 (간단 버전)
                x_overlap = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0]))
                y_overlap = max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                overlap = x_overlap * y_overlap
                area = w * h
                iou = overlap / area

                if iou < 0.5:  # 겹침이 50% 미만이면 유지
                    remaining.append(j)

            indices = np.array(remaining)

        # 결과 그리기
        for i in keep:
            pt = (boxes[i][0], boxes[i][1])
            cv2.rectangle(scene_copy, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 2)
            score = result[pt[1], pt[0]]
            cv2.putText(scene_copy, f'{score:.2f}', (pt[0], pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            match_count += 1

    print(f"임계값: {threshold}")
    print(f"검출된 객체 수: {match_count}")

    cv2.imwrite('find_all_matches.jpg', scene_copy)


def haar_cascade_demo():
    """Haar Cascade 데모"""
    print("\n" + "=" * 50)
    print("Haar Cascade 검출기")
    print("=" * 50)

    # 테스트 이미지 (얼굴 시뮬레이션)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 얼굴 형태 시뮬레이션
    cv2.ellipse(img, (200, 150), (50, 60), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (180, 130), 8, (50, 50, 50), -1)  # 눈
    cv2.circle(img, (220, 130), 8, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 170), (15, 8), 0, 0, 180, (50, 50, 50), 2)  # 입

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar Cascade 로드
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Haar Cascade 파일을 찾을 수 없습니다.")
    else:
        # 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        print(f"검출된 얼굴: {len(faces)}")

        cv2.imwrite('haar_input.jpg', img)
        cv2.imwrite('haar_result.jpg', result)

    print("\nHaar Cascade 파라미터:")
    print("  scaleFactor: 이미지 피라미드 스케일")
    print("  minNeighbors: 검출 확정 최소 이웃 수")
    print("  minSize: 최소 객체 크기")

    print("\n사용 가능한 Haar Cascade:")
    print(f"  경로: {cv2.data.haarcascades}")
    print("  - haarcascade_frontalface_default.xml")
    print("  - haarcascade_eye.xml")
    print("  - haarcascade_smile.xml")
    print("  - haarcascade_fullbody.xml")


def hog_concept_demo():
    """HOG 개념 데모"""
    print("\n" + "=" * 50)
    print("HOG (Histogram of Oriented Gradients)")
    print("=" * 50)

    # 테스트 이미지
    img = np.zeros((128, 64, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # 사람 형태 시뮬레이션
    cv2.ellipse(img, (32, 25), (12, 15), 0, 0, 360, (100, 100, 100), -1)  # 머리
    cv2.rectangle(img, (20, 40), (44, 90), (100, 100, 100), -1)  # 몸통
    cv2.rectangle(img, (18, 90), (30, 125), (100, 100, 100), -1)  # 왼쪽 다리
    cv2.rectangle(img, (34, 90), (46, 125), (100, 100, 100), -1)  # 오른쪽 다리

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG 디스크립터 계산
    # winSize: 검출 윈도우 크기
    # blockSize: 블록 크기
    # blockStride: 블록 이동 간격
    # cellSize: 셀 크기
    # nbins: 히스토그램 빈 수

    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    # HOG 특징 계산
    features = hog.compute(gray)

    print(f"입력 이미지: {gray.shape}")
    print(f"HOG 특징 벡터 크기: {features.shape}")

    print("\nHOG 특성:")
    print("  - 그래디언트 방향 히스토그램")
    print("  - 조명 변화에 강건")
    print("  - 보행자 검출에 효과적")
    print("  - SVM과 함께 사용")

    # 기본 HOG 보행자 검출기
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("\n사전 학습된 HOG+SVM 검출기:")
    print("  - cv2.HOGDescriptor_getDefaultPeopleDetector()")
    print("  - 보행자 검출용 학습된 SVM 가중치")

    cv2.imwrite('hog_input.jpg', img)


def detection_comparison():
    """검출 방법 비교"""
    print("\n" + "=" * 50)
    print("객체 검출 방법 비교")
    print("=" * 50)

    print("""
    | 방법 | 장점 | 단점 | 용도 |
    |------|------|------|------|
    | Template Matching | 간단, 빠름 | 회전/스케일 불변X | 고정 패턴 |
    | Haar Cascade | 빠름, 얼굴 특화 | 정확도 제한 | 얼굴 검출 |
    | HOG+SVM | 정확, 강건 | 느림 | 보행자 검출 |
    | 특징점 매칭 | 회전/스케일 불변 | 계산량 큼 | 객체 인식 |
    | 딥러닝 (YOLO 등) | 매우 정확 | GPU 필요 | 범용 검출 |
    """)


def main():
    """메인 함수"""
    # 템플릿 매칭
    template_matching_demo()

    # 다중 스케일
    multi_scale_template_demo()

    # 모든 매칭 찾기
    find_all_matches_demo()

    # Haar Cascade
    haar_cascade_demo()

    # HOG 개념
    hog_concept_demo()

    # 방법 비교
    detection_comparison()

    print("\n객체 검출 기초 데모 완료!")


if __name__ == '__main__':
    main()
