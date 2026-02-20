// simple_xor.c
// 간단한 XOR 암호화 데모
// 학습 목적: XOR 연산의 가역성(reversibility) 이해

#include <stdio.h>
#include <string.h>

/**
 * XOR 암호화/복호화 함수
 *
 * @param data 암호화할 데이터 (in-place 수정됨)
 * @param len 데이터 길이
 * @param key 암호화 키 (단일 문자)
 *
 * XOR의 핵심 특성:
 * - A ^ B = C
 * - C ^ B = A (같은 키로 다시 XOR하면 원본 복원)
 */
void xor_encrypt(char *data, int len, char key) {
    for (int i = 0; i < len; i++) {
        data[i] ^= key;  // XOR 연산으로 암호화/복호화
    }
}

/**
 * 바이너리 데이터를 16진수로 출력
 *
 * @param data 출력할 데이터
 * @param len 데이터 길이
 */
void print_hex(const char *data, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02X ", (unsigned char)data[i]);
    }
    printf("\n");
}

/**
 * 비트 패턴 출력 (8비트)
 *
 * @param byte 출력할 바이트
 */
void print_binary(unsigned char byte) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (byte >> i) & 1);
    }
}

int main(void) {
    char message[] = "Hello, World!";
    char key = 'K';  // 간단한 단일 문자 키 (ASCII 75)

    printf("=== XOR 암호화 데모 ===\n\n");

    // 원본 메시지 출력
    printf("원본 메시지: %s\n", message);
    printf("원본 (hex):  ");
    print_hex(message, strlen(message));
    printf("\n");

    // 첫 번째 문자의 XOR 연산 상세 설명
    printf("첫 글자 'H' XOR 'K' 연산 과정:\n");
    printf("  'H' = %d (0b", (unsigned char)message[0]);
    print_binary((unsigned char)message[0]);
    printf(")\n");
    printf("  'K' = %d (0b", (unsigned char)key);
    print_binary((unsigned char)key);
    printf(")\n");
    printf("  XOR = %d (0b", (unsigned char)(message[0] ^ key));
    print_binary((unsigned char)(message[0] ^ key));
    printf(")\n\n");

    // 암호화
    xor_encrypt(message, strlen(message), key);
    printf("암호화 완료!\n");
    printf("암호화 (hex): ");
    print_hex(message, strlen(message));

    // 암호화된 텍스트는 제어 문자가 포함될 수 있어 출력 불가
    printf("암호화 텍스트: ");
    for (int i = 0; message[i]; i++) {
        // 출력 가능한 문자만 표시
        if (message[i] >= 32 && message[i] <= 126) {
            printf("%c", message[i]);
        } else {
            printf("?");  // 제어 문자는 ? 로 표시
        }
    }
    printf("\n\n");

    // 복호화 (같은 키로 다시 XOR)
    xor_encrypt(message, strlen(message), key);
    printf("복호화 완료! (같은 키로 다시 XOR)\n");
    printf("복호화 결과: %s\n", message);
    printf("복호화 (hex): ");
    print_hex(message, strlen(message));

    // XOR의 가역성 검증
    printf("\n=== XOR 가역성 검증 ===\n");
    char test = 'A';
    printf("원본: %c (%d)\n", test, (unsigned char)test);

    test ^= key;
    printf("암호화: %c (%d)\n", test, (unsigned char)test);

    test ^= key;
    printf("복호화: %c (%d)\n", test, (unsigned char)test);
    printf("성공: 원본과 복호화 결과가 동일!\n");

    return 0;
}
