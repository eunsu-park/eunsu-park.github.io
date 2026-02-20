// file_encrypt_v2.c
// 개선된 파일 암호화 도구 (헤더 + 키 검증)
// 학습 목적: 구조체, 파일 헤더, 해시 함수, 향상된 에러 처리

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 상수 정의
#define MAGIC "XENC"          // 파일 식별용 매직 넘버
#define VERSION 1             // 파일 포맷 버전
#define BUFFER_SIZE 4096      // 4KB 버퍼
#define HEADER_SIZE 17        // 헤더 크기 (바이트)

/**
 * 암호화 파일 헤더 구조체
 *
 * 구조:
 * - magic[4]: "XENC" 매직 넘버 (파일 식별)
 * - version: 파일 포맷 버전 (1바이트)
 * - key_hash: 키의 해시값 (4바이트, 키 검증용)
 * - original_size: 원본 파일 크기 (8바이트)
 */
typedef struct {
    char magic[4];          // 매직 넘버: "XENC"
    uint8_t version;        // 버전: 1
    uint32_t key_hash;      // 키 해시 (djb2)
    uint64_t original_size; // 원본 파일 크기
} FileHeader;

// 함수 선언
void print_usage(const char *name);
uint32_t hash_key(const char *key);
void xor_buffer(unsigned char *buf, size_t len, const char *key, size_t key_len, size_t *pos);
int encrypt_file(const char *input, const char *output, const char *key);
int decrypt_file(const char *input, const char *output, const char *key);
int show_info(const char *filename);

/**
 * 간단한 해시 함수 (djb2 알고리즘)
 *
 * @param key 해시할 문자열
 * @return 32비트 해시값
 *
 * djb2는 Daniel J. Bernstein이 만든 간단하지만 효과적인 해시 함수
 * 초기값: 5381
 * 공식: hash = hash * 33 + c
 */
uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;

    while ((c = *key++)) {
        // hash * 33 + c = (hash << 5) + hash + c
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

/**
 * 사용법 출력
 */
void print_usage(const char *name) {
    printf("개선된 파일 암호화 도구 v2\n\n");
    printf("사용법:\n");
    printf("  %s encrypt <입력파일> <출력파일> <비밀번호>\n", name);
    printf("  %s decrypt <입력파일> <출력파일> <비밀번호>\n", name);
    printf("  %s info <암호화파일>\n", name);
    printf("\n예제:\n");
    printf("  %s encrypt secret.txt secret.enc mypassword\n", name);
    printf("  %s decrypt secret.enc decrypted.txt mypassword\n", name);
    printf("  %s info secret.enc\n", name);
}

/**
 * 버퍼를 XOR 암호화/복호화 (키 위치 추적)
 *
 * @param buf 처리할 버퍼
 * @param len 버퍼 길이
 * @param key 암호화 키
 * @param key_len 키 길이
 * @param pos 키의 현재 위치 (포인터, 상태 유지)
 *
 * pos 매개변수로 여러 버퍼에 걸쳐 키 위치를 연속적으로 추적
 * 이렇게 하면 파일을 여러 블록으로 나눠 읽어도 키 패턴이 일관됨
 */
void xor_buffer(unsigned char *buf, size_t len, const char *key, size_t key_len, size_t *pos) {
    for (size_t i = 0; i < len; i++) {
        buf[i] ^= key[*pos % key_len];
        (*pos)++;  // 키 위치 증가
    }
}

/**
 * 파일 암호화 (헤더 포함)
 *
 * @param input 입력 파일 경로
 * @param output 출력 파일 경로
 * @param key 암호화 키
 * @return 성공 시 0, 실패 시 1
 */
int encrypt_file(const char *input, const char *output, const char *key) {
    // 입력 파일 열기
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("입력 파일 열기 실패");
        return 1;
    }

    // 원본 파일 크기 확인
    fseek(fin, 0, SEEK_END);
    uint64_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    // 출력 파일 열기
    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("출력 파일 열기 실패");
        fclose(fin);
        return 1;
    }

    // 헤더 구조체 초기화
    FileHeader header;
    memcpy(header.magic, MAGIC, 4);     // "XENC" 복사
    header.version = VERSION;            // 버전 1
    header.key_hash = hash_key(key);     // 키의 해시값 저장
    header.original_size = file_size;    // 원본 크기 저장

    // 헤더를 파일에 기록
    fwrite(&header, sizeof(FileHeader), 1, fout);

    // 데이터 암호화
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;  // 키 위치 추적

    printf("암호화 중...\n");
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);

        // 간단한 진행률 표시
        printf(".");
        fflush(stdout);
    }
    printf("\n");

    fclose(fin);
    fclose(fout);

    printf("암호화 완료: %s -> %s\n", input, output);
    printf("원본 크기: %llu 바이트\n", (unsigned long long)file_size);
    printf("키 해시: 0x%08X\n", header.key_hash);

    return 0;
}

/**
 * 파일 복호화 (헤더 검증 포함)
 *
 * @param input 입력 파일 경로 (암호화된 파일)
 * @param output 출력 파일 경로 (복호화된 파일)
 * @param key 복호화 키
 * @return 성공 시 0, 실패 시 1
 */
int decrypt_file(const char *input, const char *output, const char *key) {
    // 암호화된 파일 열기
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("입력 파일 열기 실패");
        return 1;
    }

    // 헤더 읽기
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fin) != 1) {
        fprintf(stderr, "오류: 잘못된 암호화 파일 (헤더 읽기 실패)\n");
        fclose(fin);
        return 1;
    }

    // 매직 넘버 확인 (파일이 XENC 형식인지 검증)
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        fprintf(stderr, "오류: 유효한 암호화 파일이 아닙니다 (매직 넘버 불일치)\n");
        fprintf(stderr, "기대값: %.4s, 실제값: %.4s\n", MAGIC, header.magic);
        fclose(fin);
        return 1;
    }

    // 버전 확인
    if (header.version != VERSION) {
        fprintf(stderr, "경고: 파일 버전이 다릅니다 (기대: %d, 실제: %d)\n",
                VERSION, header.version);
    }

    // 키 검증 (해시 비교)
    uint32_t input_key_hash = hash_key(key);
    if (header.key_hash != input_key_hash) {
        fprintf(stderr, "오류: 잘못된 비밀번호\n");
        fprintf(stderr, "기대 해시: 0x%08X, 입력 해시: 0x%08X\n",
                header.key_hash, input_key_hash);
        fclose(fin);
        return 1;
    }

    // 출력 파일 열기
    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("출력 파일 열기 실패");
        fclose(fin);
        return 1;
    }

    // 데이터 복호화
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;  // 키 위치 추적

    printf("복호화 중...\n");
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);

        // 진행률 표시
        printf(".");
        fflush(stdout);
    }
    printf("\n");

    fclose(fin);
    fclose(fout);

    printf("복호화 완료: %s -> %s\n", input, output);
    printf("원본 크기: %llu 바이트\n", (unsigned long long)header.original_size);

    return 0;
}

/**
 * 암호화 파일 정보 표시
 *
 * @param filename 암호화 파일 경로
 * @return 성공 시 0, 실패 시 1
 */
int show_info(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("파일 열기 실패");
        return 1;
    }

    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        fprintf(stderr, "오류: 헤더를 읽을 수 없습니다\n");
        fclose(fp);
        return 1;
    }

    // 파일 전체 크기 확인
    fseek(fp, 0, SEEK_END);
    long total_size = ftell(fp);
    fclose(fp);

    // 매직 넘버 확인
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        printf("암호화 파일이 아닙니다 (XENC 매직 넘버 없음)\n");
        return 1;
    }

    // 정보 출력
    printf("=== 암호화 파일 정보 ===\n");
    printf("매직 넘버: %.4s\n", header.magic);
    printf("버전: %d\n", header.version);
    printf("키 해시: 0x%08X\n", header.key_hash);
    printf("원본 크기: %llu 바이트\n", (unsigned long long)header.original_size);
    printf("파일 크기: %ld 바이트\n", total_size);
    printf("헤더 크기: %lu 바이트\n", sizeof(FileHeader));
    printf("암호화 데이터: %ld 바이트\n", total_size - (long)sizeof(FileHeader));

    return 0;
}

/**
 * 메인 함수
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // 명령 분기
    if (strcmp(argv[1], "encrypt") == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        return encrypt_file(argv[2], argv[3], argv[4]);
    }
    else if (strcmp(argv[1], "decrypt") == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        return decrypt_file(argv[2], argv[3], argv[4]);
    }
    else if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            print_usage(argv[0]);
            return 1;
        }
        return show_info(argv[2]);
    }
    else {
        fprintf(stderr, "오류: 알 수 없는 명령 '%s'\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
