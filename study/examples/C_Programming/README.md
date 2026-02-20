# C 프로그래밍 예제 코드

이 폴더에는 C 프로그래밍 학습 문서의 모든 예제 코드가 포함되어 있습니다.

## 디렉토리 구조

```
examples/
├── 02_calculator/          # 계산기
├── 03_number_guess/        # 숫자 맞추기 게임
├── 04_address_book/        # 주소록
├── 05_dynamic_array/       # 동적 배열
├── 06_linked_list/         # 연결 리스트
├── 07_file_crypto/         # 파일 암호화
├── 08_stack_queue/         # 스택과 큐
├── 09_hash_table/          # 해시 테이블
├── 10_snake_game/          # 뱀 게임
├── 11_minishell/           # 미니 쉘
├── 12_multithread/         # 멀티스레드
├── 13_embedded_basic/      # 임베디드 기초 (Arduino)
├── 14_bit_operations/      # 비트 연산
├── 15_gpio_control/        # GPIO 제어 (Arduino)
└── 16_serial_comm/         # 시리얼 통신 (Arduino)
```

## 컴파일 방법

### C 프로그램 (데스크톱)

```bash
# 기본 컴파일
gcc program.c -o program

# 경고 포함
gcc -Wall -Wextra program.c -o program

# 디버그 정보 포함
gcc -g program.c -o program

# 최적화
gcc -O2 program.c -o program
```

### 멀티스레드 프로그램

```bash
# Linux
gcc program.c -o program -pthread

# macOS
gcc program.c -o program -lpthread
```

### Arduino 프로그램

Arduino 프로그램(.ino)은 다음 방법으로 실행:

1. **Arduino IDE**
   - 파일 열기
   - 보드 선택 (Tools → Board → Arduino Uno)
   - 업로드 버튼 클릭

2. **Wokwi 시뮬레이터** (권장)
   - https://wokwi.com 접속
   - New Project → Arduino Uno
   - 코드 복사/붙여넣기
   - Start Simulation

3. **PlatformIO (VS Code)**
   ```bash
   pio run
   pio run --target upload
   ```

## 각 프로젝트 설명

| 프로젝트 | 난이도 | 주요 개념 |
|----------|--------|-----------|
| 02. 계산기 | ⭐ | 함수, switch-case, scanf |
| 03. 숫자 맞추기 | ⭐ | 반복문, 랜덤, 조건문 |
| 04. 주소록 | ⭐⭐ | 구조체, 배열, 파일 I/O |
| 05. 동적 배열 | ⭐⭐ | malloc, realloc, free |
| 06. 연결 리스트 | ⭐⭐⭐ | 포인터, 동적 자료구조 |
| 07. 파일 암호화 | ⭐⭐ | 파일 처리, 비트 연산 |
| 08. 스택/큐 | ⭐⭐ | 자료구조, LIFO/FIFO |
| 09. 해시 테이블 | ⭐⭐⭐ | 해싱, 충돌 처리 |
| 10. 뱀 게임 | ⭐⭐⭐ | 터미널 제어, 게임 루프 |
| 11. 미니 쉘 | ⭐⭐⭐⭐ | fork, exec, 파이프 |
| 12. 멀티스레드 | ⭐⭐⭐⭐ | pthread, 동기화 |
| 13. 임베디드 기초 | ⭐ | Arduino, GPIO |
| 14. 비트 연산 | ⭐⭐ | 비트 마스킹, 레지스터 |
| 15. GPIO 제어 | ⭐⭐ | LED, 버튼, 디바운싱 |
| 16. 시리얼 통신 | ⭐⭐ | UART, 명령어 파싱 |

## 학습 순서

### 초급
1. 계산기
2. 숫자 맞추기
3. 주소록

### 중급
4. 동적 배열
5. 연결 리스트
6. 파일 암호화
7. 스택과 큐
8. 해시 테이블

### 고급
9. 뱀 게임
10. 미니 쉘
11. 멀티스레드

### 임베디드 (Arduino)
12. 임베디드 기초
13. 비트 연산
14. GPIO 제어
15. 시리얼 통신

## 실행 예시

### 계산기
```bash
cd 02_calculator
gcc calculator.c -o calculator
./calculator
```

### 멀티스레드
```bash
cd 12_multithread
gcc thread_basic.c -o thread_basic -pthread
./thread_basic
```

### Arduino (Wokwi)
1. 코드 복사
2. https://wokwi.com 에서 새 프로젝트
3. 붙여넣기 후 실행

## 문제 해결

### 컴파일 오류
- `undefined reference to 'pthread_create'`: `-pthread` 플래그 추가
- `implicit declaration of function`: 헤더 파일 추가
- `permission denied`: `chmod +x program`

### 실행 오류
- Segmentation fault: 포인터 확인, valgrind 사용
- Bus error: 잘못된 메모리 접근
- Memory leak: valgrind로 확인

## 추가 자료

- [C Reference](https://en.cppreference.com/w/c)
- [Arduino Reference](https://www.arduino.cc/reference/en/)
- [Wokwi Simulator](https://wokwi.com)
