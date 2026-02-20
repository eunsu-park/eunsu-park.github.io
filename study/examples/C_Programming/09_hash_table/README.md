# Hash Table Project (해시 테이블 프로젝트)

다양한 해시 함수와 충돌 해결 기법을 구현한 해시 테이블 프로젝트입니다.

## 파일 구성

### 1. hash_functions.c
다양한 해시 함수의 비교 및 분석 도구

**구현된 해시 함수:**
- `hash_simple` - 단순 합산 (충돌 많음, 나쁜 예)
- `hash_djb2` - Daniel J. Bernstein (추천)
- `hash_sdbm` - sdbm 데이터베이스 해시
- `hash_fnv1a` - Fowler-Noll-Vo 1a

**기능:**
- 해시 값 비교 출력
- 충돌 횟수 및 충돌률 분석
- 분포 균일성 분석 (분산 계산)
- 해시 분포 시각화

### 2. hash_chaining.c
체이닝(Separate Chaining)을 이용한 해시 테이블 구현

**특징:**
- 충돌 시 연결 리스트로 저장
- 테이블 크기 제한 없음
- 삽입/삭제 간단

**구현 기능:**
- `ht_create()` - 해시 테이블 생성
- `ht_set()` - 삽입/수정
- `ht_get()` - 검색
- `ht_delete()` - 삭제
- `ht_print()` - 테이블 출력
- `ht_get_statistics()` - 통계 수집
- 체인 길이 분포 시각화

### 3. hash_linear_probing.c
선형 탐사(Linear Probing)를 이용한 오픈 어드레싱 해시 테이블

**특징:**
- 충돌 시 다음 빈 슬롯 탐색
- 캐시 효율 좋음
- DELETED 상태로 삭제 처리

**구현 기능:**
- 세 가지 슬롯 상태 (EMPTY, OCCUPIED, DELETED)
- 선형 탐사 알고리즘
- 클러스터링 분석
- 로드 팩터별 성능 비교
- 클러스터 시각화

### 4. dictionary.c
해시 테이블을 활용한 실용적인 사전 프로그램

**주요 기능:**
- 단어 추가/검색/삭제
- 전체 목록 출력
- 파일 저장/불러오기 (dictionary.txt)
- 검색 제안 (부분 일치)
- 검색 통계 및 인기 단어 Top 10
- 대소문자 구분 없는 검색

**데이터 구조:**
- 체이닝 방식
- 테이블 크기: 1000
- 검색 횟수 추적

## 컴파일 및 실행

### 개별 컴파일
```bash
gcc -Wall -Wextra -std=c11 -o hash_functions hash_functions.c
gcc -Wall -Wextra -std=c11 -o hash_chaining hash_chaining.c
gcc -Wall -Wextra -std=c11 -o hash_linear_probing hash_linear_probing.c
gcc -Wall -Wextra -std=c11 -o dictionary dictionary.c
```

### Makefile 사용
```bash
make                # 모든 프로그램 컴파일
make hash_functions # 특정 프로그램만 컴파일
make clean          # 생성된 파일 삭제
make run_dict       # 사전 프로그램 실행
```

### 실행
```bash
./hash_functions        # 해시 함수 비교
./hash_chaining         # 체이닝 테스트
./hash_linear_probing   # 선형 탐사 테스트
./dictionary            # 사전 프로그램
```

## 학습 포인트

### 해시 함수 선택
- **djb2**: 일반적인 용도, 균형있는 성능
- **FNV-1a**: 빠른 속도 필요 시
- **sdbm**: 데이터베이스 용도
- **Simple은 사용하지 말 것** (충돌률 높음)

### 충돌 해결 방법 비교

| 비교 항목 | 체이닝 | 오픈 어드레싱 |
|-----------|--------|---------------|
| 메모리 | 동적 할당 | 고정 크기 |
| 삭제 | 간단 | DELETED 표시 필요 |
| 캐시 효율 | 불리 | 유리 |
| 로드 팩터 | >1 가능 | <1 필수 |
| 구현 복잡도 | 낮음 | 중간 |

### 시간 복잡도

| 연산 | 평균 | 최악 |
|------|------|------|
| 삽입 | O(1) | O(n) |
| 검색 | O(1) | O(n) |
| 삭제 | O(1) | O(n) |

### 성능 최적화 팁
1. 로드 팩터 0.7 이하 유지
2. 좋은 해시 함수 선택 (djb2 추천)
3. 테이블 크기는 소수(prime number) 사용 권장
4. 체이닝 vs 오픈 어드레싱 선택은 용도에 따라

## 예제 출력

### hash_functions 실행 결과
```
=== 해시 함수 비교 ===

Key          | Simple | djb2 | sdbm | fnv1a
-------------|--------|------|------|------
apple        |     30 |   43 |   58 |    67
banana       |      9 |   42 |   49 |    52

=== 충돌 분석 ===
Simple       |        14 | 28.0%
djb2         |         6 | 12.0%  ← 최소 충돌
```

### dictionary 사용 예
```
선택: 1
단어: programming
뜻: 프로그래밍; 컴퓨터에 명령을 작성하는 작업
✓ 'programming' 추가됨

선택: 2
검색할 단어: prog
'prog'로 시작하는 단어:
  - programming
총 1개 발견
```

## 확장 아이디어

1. **동적 크기 조절**: 로드 팩터가 0.7 초과 시 테이블 크기 2배 확장
2. **이중 해싱**: 두 번째 해시 함수로 탐사 간격 결정
3. **사전 기능 추가**:
   - 단어 예문
   - 발음 기호
   - 동의어/반의어
4. **성능 측정**: 각 연산의 실제 수행 시간 측정
5. **멀티스레드**: 읽기 동시성 지원

## 참고 자료

- 해시 함수: [djb2, sdbm, FNV-1a 알고리즘](http://www.cse.yorku.ca/~oz/hash.html)
- 충돌 해결: Cormen et al., "Introduction to Algorithms"
- 실전 활용: Python `dict`, Java `HashMap`, C++ `unordered_map`
