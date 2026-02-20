# Snake Game Examples

터미널 기반 뱀 게임 프로젝트 예제 파일들입니다.

## 파일 구성

### 1. `ansi_demo.c`
ANSI escape codes 시연 프로그램
- 커서 이동, 색상, 박스 그리기 등을 보여줍니다.

**컴파일 및 실행:**
```bash
gcc -o ansi_demo ansi_demo.c
./ansi_demo
```

### 2. `input_demo.c`
비동기 키보드 입력 시연 프로그램
- termios를 사용한 non-blocking 입력 처리
- 방향키와 WASD 키 지원

**컴파일 및 실행:**
```bash
gcc -o input_demo input_demo.c
./input_demo
```

**조작법:**
- 방향키 또는 WASD: 이동
- 스페이스: 점프
- Q: 종료

### 3. `snake_types.h`
뱀 게임 데이터 구조 정의
- 게임에 필요한 모든 타입과 상수 정의
- 다른 파일에서 include하여 사용

### 4. `snake.c`
뱀 게임 핵심 로직 구현
- 뱀 생성/해제, 이동, 충돌 검사 등
- `snake_types.h`를 사용하는 모듈형 구현

**컴파일 (단독 실행 불가):**
```bash
# 다른 파일과 함께 링크해야 함
gcc -c snake.c -o snake.o
```

### 5. `snake_game.c`
완성된 뱀 게임 (ANSI escape codes 사용)
- 독립적으로 동작하는 완전한 게임
- 외부 라이브러리 필요 없음

**컴파일 및 실행:**
```bash
gcc -o snake_game snake_game.c
./snake_game
```

**조작법:**
- 방향키 또는 WASD: 뱀 이동
- P: 일시정지
- Q: 종료
- R: 재시작 (게임 오버 후)

**기능:**
- 점수 시스템
- 속도 증가 (음식 먹을수록 빨라짐)
- 최고 점수 저장 (`.snake_highscore` 파일)
- 일시정지
- 색상 표시

### 6. `snake_ncurses.c`
NCurses 라이브러리를 사용한 향상된 버전

**⚠️ 이 파일은 ncurses 라이브러리가 필요합니다!**

**라이브러리 설치:**
```bash
# macOS
brew install ncurses

# Ubuntu/Debian
sudo apt install libncurses5-dev

# Fedora/RHEL
sudo dnf install ncurses-devel
```

**컴파일 및 실행:**
```bash
# macOS
gcc -o snake_ncurses snake_ncurses.c -lncurses

# Linux
gcc -o snake_ncurses snake_ncurses.c -lncurses

# 실행
./snake_ncurses
```

**조작법:**
- 방향키 또는 WASD: 뱀 이동
- P: 일시정지
- Q: 종료
- R: 재시작 (게임 오버 후)

**ncurses 버전의 장점:**
- 더 깔끔한 화면 처리
- 자동 버퍼링 및 깜빡임 방지
- 표준 박스 문자 사용
- 더 나은 색상 관리

## 학습 순서

1. **`ansi_demo.c`** - ANSI escape codes 이해
2. **`input_demo.c`** - 비동기 입력 처리 이해
3. **`snake_types.h`** - 게임 데이터 구조 파악
4. **`snake.c`** - 게임 로직 구현 학습
5. **`snake_game.c`** - 완성된 게임 분석
6. **`snake_ncurses.c`** - ncurses 라이브러리 활용 (선택)

## 개발 환경

- C11 표준
- POSIX 호환 시스템 (Linux, macOS, BSD)
- 터미널: UTF-8 지원 필요
- 컴파일러: GCC 또는 Clang

## 참고 자료

이 예제는 다음 학습 자료를 기반으로 합니다:
- `/content/ko/C_Programming/11_Project_Snake_Game.md`

## 문제 해결

### 문제: 박스 문자가 깨져 보임
**해결:** 터미널이 UTF-8을 지원하는지 확인
```bash
echo $LANG
# 출력 예: en_US.UTF-8 또는 ko_KR.UTF-8
```

### 문제: 키 입력이 작동하지 않음
**해결:** 터미널이 ANSI escape sequences를 지원하는지 확인

### 문제: ncurses 링크 오류
**해결:** ncurses 개발 패키지 설치 확인
```bash
# macOS
brew list ncurses

# Ubuntu
dpkg -l | grep libncurses
```

## 라이선스

교육 목적으로 자유롭게 사용 가능합니다.
