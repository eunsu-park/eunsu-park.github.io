// file_encrypt.c
// 파일 암호화 도구 (XOR 기반)
// 학습 목적: 바이트 단위 파일 I/O, 명령줄 인자 처리

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4096  // 4KB 버퍼 크기

// 함수 선언
void print_usage(const char *program_name);
int encrypt_file(const char *input_file, const char *output_file, const char *key);
int decrypt_file(const char *input_file, const char *output_file, const char *key);
void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len);

/**
 * 메인 함수 - 명령줄 인자 파싱
 *
 * @param argc 인자 개수
 * @param argv 인자 배열
 *
 * 사용법:
 *   ./file_encrypt -e input.txt output.enc mypassword
 *   ./file_encrypt -d output.enc decrypted.txt mypassword
 */
int main(int argc, char *argv[]) {
    // 인자 개수 확인 (프로그램명 + 모드 + 입력 + 출력 + 키 = 5개)
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];         // -e 또는 -d
    const char *input_file = argv[2];   // 입력 파일명
    const char *output_file = argv[3];  // 출력 파일명
    const char *key = argv[4];          // 암호화 키

    // 빈 키 검증
    if (strlen(key) == 0) {
        fprintf(stderr, "오류: 키는 비어있을 수 없습니다\n");
        return 1;
    }

    int result;
    if (strcmp(mode, "-e") == 0 || strcmp(mode, "--encrypt") == 0) {
        // 암호화 모드
        result = encrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("암호화 성공: %s -> %s\n", input_file, output_file);
        }
    } else if (strcmp(mode, "-d") == 0 || strcmp(mode, "--decrypt") == 0) {
        // 복호화 모드
        result = decrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("복호화 성공: %s -> %s\n", input_file, output_file);
        }
    } else {
        fprintf(stderr, "오류: 알 수 없는 모드 '%s'\n", mode);
        print_usage(argv[0]);
        return 1;
    }

    return result;
}

/**
 * 사용법 출력
 *
 * @param program_name 프로그램 이름 (argv[0])
 */
void print_usage(const char *program_name) {
    printf("파일 암호화 도구 (XOR)\n\n");
    printf("사용법:\n");
    printf("  %s -e <입력파일> <출력파일> <키>  파일 암호화\n", program_name);
    printf("  %s -d <입력파일> <출력파일> <키>  파일 복호화\n", program_name);
    printf("\n옵션:\n");
    printf("  -e, --encrypt  암호화 모드\n");
    printf("  -d, --decrypt  복호화 모드\n");
    printf("\n예제:\n");
    printf("  %s -e secret.txt secret.enc mypassword\n", program_name);
    printf("  %s -d secret.enc secret.txt mypassword\n", program_name);
}

/**
 * 버퍼를 XOR 암호화/복호화
 *
 * @param buffer 처리할 버퍼
 * @param len 버퍼 길이
 * @param key 암호화 키 (문자열)
 * @param key_len 키 길이
 *
 * 키가 데이터보다 짧으면 반복해서 사용 (modulo 연산)
 */
void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len) {
    for (int i = 0; i < len; i++) {
        // 키를 순환하며 XOR 적용
        buffer[i] ^= key[i % key_len];
    }
}

/**
 * 파일 암호화 함수
 *
 * @param input_file 입력 파일 경로
 * @param output_file 출력 파일 경로
 * @param key 암호화 키
 * @return 성공 시 0, 실패 시 1
 */
int encrypt_file(const char *input_file, const char *output_file, const char *key) {
    // 입력 파일 열기 (바이너리 읽기 모드)
    FILE *fin = fopen(input_file, "rb");
    if (fin == NULL) {
        perror("입력 파일 열기 실패");
        return 1;
    }

    // 출력 파일 열기 (바이너리 쓰기 모드)
    FILE *fout = fopen(output_file, "wb");
    if (fout == NULL) {
        perror("출력 파일 열기 실패");
        fclose(fin);
        return 1;
    }

    // 버퍼 준비
    unsigned char buffer[BUFFER_SIZE];
    int key_len = strlen(key);
    size_t bytes_read;

    // 파일을 버퍼 크기만큼씩 읽어서 처리
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        // XOR 암호화 적용
        xor_buffer(buffer, bytes_read, key, key_len);

        // 암호화된 데이터 쓰기
        fwrite(buffer, 1, bytes_read, fout);
    }

    // 파일 닫기
    fclose(fin);
    fclose(fout);

    return 0;
}

/**
 * 파일 복호화 함수
 *
 * @param input_file 입력 파일 경로
 * @param output_file 출력 파일 경로
 * @param key 암호화 키
 * @return 성공 시 0, 실패 시 1
 *
 * XOR 암호화의 특성상 암호화와 복호화 과정이 동일함
 * (A ^ K = B, B ^ K = A)
 */
int decrypt_file(const char *input_file, const char *output_file, const char *key) {
    // XOR 암호화는 암호화와 복호화가 동일한 연산
    return encrypt_file(input_file, output_file, key);
}
