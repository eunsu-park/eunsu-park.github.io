# Quick Start Guide - File Encryption Examples

## Build & Run (빠른 시작)

```bash
# 모든 프로그램 컴파일
make

# 모든 테스트 실행
make test

# 정리
make clean
```

## 개별 실행 예제

### 1. XOR 데모 보기
```bash
gcc -Wall -Wextra -std=c11 simple_xor.c -o simple_xor
./simple_xor
```

### 2. 파일 암호화 (기본 버전)
```bash
gcc -Wall -Wextra -std=c11 file_encrypt.c -o file_encrypt

# 사용법
./file_encrypt -e <입력> <출력> <키>    # 암호화
./file_encrypt -d <입력> <출력> <키>    # 복호화

# 예제
echo "Secret data" > secret.txt
./file_encrypt -e secret.txt secret.enc mypassword
./file_encrypt -d secret.enc decrypted.txt mypassword
cat decrypted.txt
```

### 3. 파일 암호화 v2 (헤더 + 검증)
```bash
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o file_encrypt_v2

# 사용법
./file_encrypt_v2 encrypt <입력> <출력> <비밀번호>
./file_encrypt_v2 decrypt <입력> <출력> <비밀번호>
./file_encrypt_v2 info <암호화파일>

# 예제
echo "Top secret!" > data.txt
./file_encrypt_v2 encrypt data.txt data.enc strongpass
./file_encrypt_v2 info data.enc
./file_encrypt_v2 decrypt data.enc restored.txt strongpass
diff data.txt restored.txt  # 동일해야 함
```

## 주요 차이점

| 기능 | file_encrypt | file_encrypt_v2 |
|------|--------------|-----------------|
| 인터페이스 | `-e` / `-d` 옵션 | `encrypt` / `decrypt` 명령 |
| 파일 헤더 | ✗ | ✓ (XENC 매직 넘버) |
| 키 검증 | ✗ | ✓ (해시 비교) |
| 메타데이터 | ✗ | ✓ (원본 크기, 버전) |
| 파일 정보 | ✗ | ✓ (`info` 명령) |
| 진행률 표시 | ✗ | ✓ |

## 학습 순서 추천

1. `simple_xor.c` - XOR 기본 원리 이해
2. `file_encrypt.c` - 파일 I/O 및 기본 구조
3. `file_encrypt_v2.c` - 헤더, 검증, 고급 기능

## 보안 경고

⚠️ **학습 전용 - 실제 보안 용도로 사용 금지**

실제 암호화가 필요한 경우:
- `openssl enc -aes-256-cbc -in file -out file.enc`
- GPG (GNU Privacy Guard)
- libsodium 라이브러리
