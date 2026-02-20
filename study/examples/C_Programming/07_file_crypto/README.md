# 파일 암호화 예제 (File Encryption Examples)

XOR 기반 파일 암호화 도구 구현 예제

## 파일 목록

### 1. `simple_xor.c`
XOR 암호화의 기본 원리를 보여주는 간단한 데모

**학습 내용:**
- XOR 연산의 가역성 (A ^ B ^ B = A)
- 비트 연산 기초
- 16진수 출력
- 바이너리 패턴 출력

**컴파일 및 실행:**
```bash
gcc -Wall -Wextra -std=c11 simple_xor.c -o simple_xor
./simple_xor
```

**출력 예시:**
```
=== XOR 암호화 데모 ===

원본 메시지: Hello, World!
원본 (hex):  48 65 6C 6C 6F 2C 20 57 6F 72 6C 64 21

첫 글자 'H' XOR 'K' 연산 과정:
  'H' = 72 (0b01001000)
  'K' = 75 (0b01001011)
  XOR = 3 (0b00000011)

암호화 완료!
복호화 결과: Hello, World!
```

---

### 2. `file_encrypt.c`
실용적인 파일 암호화 도구 (기본 버전)

**학습 내용:**
- 바이트 단위 파일 I/O (`fread`, `fwrite`)
- 명령줄 인자 파싱 (`argc`, `argv`)
- 버퍼링된 파일 처리 (4KB 버퍼)
- 에러 처리 (`perror`)
- 키 순환 적용 (modulo 연산)

**컴파일:**
```bash
gcc -Wall -Wextra -std=c11 file_encrypt.c -o file_encrypt
```

**사용법:**
```bash
# 암호화
./file_encrypt -e input.txt output.enc mypassword

# 복호화
./file_encrypt -d output.enc decrypted.txt mypassword
```

**특징:**
- 간단하고 직관적인 인터페이스
- XOR 특성상 암호화/복호화 동일 연산
- 모든 파일 타입 지원 (텍스트, 바이너리)

---

### 3. `file_encrypt_v2.c`
향상된 파일 암호화 도구 (헤더 + 검증)

**학습 내용:**
- 구조체 활용 (`FileHeader`)
- 파일 매직 넘버 검증
- 해시 함수 구현 (djb2 알고리즘)
- 키 검증 (비밀번호 확인)
- 파일 메타데이터 저장
- 진행률 표시
- 고정 크기 정수 타입 (`uint8_t`, `uint32_t`, `uint64_t`)

**컴파일:**
```bash
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o file_encrypt_v2
```

**사용법:**
```bash
# 암호화
./file_encrypt_v2 encrypt secret.txt secret.enc mypassword

# 파일 정보 확인
./file_encrypt_v2 info secret.enc

# 복호화
./file_encrypt_v2 decrypt secret.enc decrypted.txt mypassword
```

**파일 헤더 구조:**
```
┌─────────────────────────────────────────┐
│  Magic Number (4 bytes): "XENC"         │
│  Version (1 byte): 1                    │
│  Key Hash (4 bytes): 검증용             │
│  Original Size (8 bytes): 원본 크기     │
├─────────────────────────────────────────┤
│         암호화된 데이터                 │
└─────────────────────────────────────────┘
```

**출력 예시:**
```bash
$ ./file_encrypt_v2 encrypt test.txt test.enc mypass
암호화 중...
.
암호화 완료: test.txt -> test.enc
원본 크기: 38 바이트
키 해시: 0x6CBFD0D9

$ ./file_encrypt_v2 info test.enc
=== 암호화 파일 정보 ===
매직 넘버: XENC
버전: 1
키 해시: 0x6CBFD0D9
원본 크기: 38 바이트
파일 크기: 62 바이트
헤더 크기: 24 바이트
암호화 데이터: 38 바이트

$ ./file_encrypt_v2 decrypt test.enc out.txt wrongpass
오류: 잘못된 비밀번호
기대 해시: 0x6CBFD0D9, 입력 해시: 0x289A5245
```

---

## 핵심 개념

### XOR 암호화 원리
```
원리: A ^ K = C, C ^ K = A

예시:
  원본: 'H' (72) = 0b01001000
  키:   'K' (75) = 0b01001011
  암호: XOR     = 0b00000011 (3)
  복호: 3 ^ 75  = 0b01001000 (72 = 'H')
```

### djb2 해시 알고리즘
```c
uint32_t hash = 5381;
while ((c = *str++)) {
    hash = hash * 33 + c;
}
```
- 빠르고 효과적인 문자열 해시
- 충돌 확률 낮음
- 키 검증용으로 적합

---

## 보안 주의사항

⚠️ **교육 목적 전용**

이 예제는 학습 목적으로 제작되었습니다. 실제 보안이 필요한 경우:

1. **XOR 암호화의 취약점:**
   - 키 반복 사용 시 패턴 노출
   - 알려진 평문 공격(Known-plaintext attack)에 취약
   - 키 길이가 짧으면 브루트포스 공격 가능

2. **실제 사용 권장사항:**
   - AES-256 (대칭 암호화)
   - RSA (비대칭 암호화)
   - OpenSSL 라이브러리 사용
   - 키 스트레칭 (PBKDF2, bcrypt)
   - 솔트(Salt) 추가

3. **이 구현의 한계:**
   - 키 스트레칭 없음
   - 솔트 미사용
   - 무결성 검증(MAC) 없음
   - 재생 공격 방지 없음

---

## 테스트 예제

```bash
# 1. 기본 테스트
echo "Hello, World!" > test.txt
./file_encrypt -e test.txt test.enc mykey
./file_encrypt -d test.enc out.txt mykey
diff test.txt out.txt  # 동일해야 함

# 2. 바이너리 파일 테스트
./file_encrypt_v2 encrypt /bin/ls ls.enc strongpass
./file_encrypt_v2 decrypt ls.enc ls.dec strongpass
diff /bin/ls ls.dec  # 동일해야 함

# 3. 잘못된 키 테스트
./file_encrypt_v2 decrypt test.enc wrong.txt wrongkey
# 출력: 오류: 잘못된 비밀번호

# 4. 대용량 파일 테스트 (10MB)
dd if=/dev/urandom of=large.bin bs=1M count=10
time ./file_encrypt_v2 encrypt large.bin large.enc mypass
./file_encrypt_v2 info large.enc
```

---

## 컴파일 옵션 설명

```bash
gcc -Wall -Wextra -std=c11 file.c -o program
```

- `-Wall`: 기본 경고 활성화
- `-Wextra`: 추가 경고 활성화
- `-std=c11`: C11 표준 사용
- `-o program`: 출력 파일명 지정

---

## 학습 체크리스트

- [ ] XOR 연산의 가역성 이해
- [ ] 비트 연산자 사용법 (`^`, `&`, `|`, `~`, `<<`, `>>`)
- [ ] 바이트 단위 파일 I/O (`fread`, `fwrite`)
- [ ] 명령줄 인자 파싱 (`argc`, `argv`)
- [ ] 구조체를 활용한 파일 헤더 설계
- [ ] 해시 함수 구현 (djb2)
- [ ] 에러 처리 및 검증 로직
- [ ] 버퍼링을 통한 효율적인 파일 처리

---

## 참고 자료

- [C11 Standard](https://en.cppreference.com/w/c/11)
- [XOR Cipher - Wikipedia](https://en.wikipedia.org/wiki/XOR_cipher)
- [djb2 Hash Function](http://www.cse.yorku.ca/~oz/hash.html)
- [OpenSSL Documentation](https://www.openssl.org/docs/)

---

## 관련 학습 자료

- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/08_Project_File_Encryption.md`
- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/14_Bit_Operations.md`
