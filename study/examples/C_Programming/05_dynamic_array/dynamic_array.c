// dynamic_array.c
// 동적 배열 구현

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} DynamicArray;

// 동적 배열 생성
DynamicArray* array_create(size_t initial_capacity) {
    DynamicArray* arr = malloc(sizeof(DynamicArray));
    if (!arr) return NULL;

    arr->data = malloc(initial_capacity * sizeof(int));
    if (!arr->data) {
        free(arr);
        return NULL;
    }

    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

// 요소 추가
int array_push(DynamicArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        // 용량 2배로 확장
        size_t new_capacity = arr->capacity * 2;
        int* new_data = realloc(arr->data, new_capacity * sizeof(int));

        if (!new_data) return 0;  // 실패

        arr->data = new_data;
        arr->capacity = new_capacity;

        printf("용량 확장: %zu -> %zu\n", arr->capacity / 2, arr->capacity);
    }

    arr->data[arr->size++] = value;
    return 1;  // 성공
}

// 요소 제거
int array_pop(DynamicArray* arr, int* value) {
    if (arr->size == 0) return 0;  // 빈 배열

    *value = arr->data[--arr->size];
    return 1;
}

// 특정 인덱스 값 가져오기
int array_get(DynamicArray* arr, size_t index, int* value) {
    if (index >= arr->size) return 0;

    *value = arr->data[index];
    return 1;
}

// 특정 인덱스 값 설정
int array_set(DynamicArray* arr, size_t index, int value) {
    if (index >= arr->size) return 0;

    arr->data[index] = value;
    return 1;
}

// 배열 출력
void array_print(DynamicArray* arr) {
    printf("[");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) printf(", ");
    }
    printf("]\n");
}

// 메모리 해제
void array_destroy(DynamicArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int main(void) {
    DynamicArray* arr = array_create(2);

    printf("=== 동적 배열 테스트 ===\n\n");

    // 요소 추가
    printf("요소 추가: 10, 20, 30, 40, 50\n");
    array_push(arr, 10);
    array_push(arr, 20);
    array_push(arr, 30);
    array_push(arr, 40);
    array_push(arr, 50);

    printf("배열: ");
    array_print(arr);
    printf("크기: %zu, 용량: %zu\n\n", arr->size, arr->capacity);

    // 요소 제거
    int value;
    array_pop(arr, &value);
    printf("제거된 값: %d\n", value);
    printf("배열: ");
    array_print(arr);
    printf("\n");

    // 특정 인덱스 값 변경
    array_set(arr, 1, 999);
    printf("인덱스 1을 999로 변경\n");
    printf("배열: ");
    array_print(arr);
    printf("\n");

    // 값 가져오기
    array_get(arr, 2, &value);
    printf("인덱스 2의 값: %d\n", value);

    // 메모리 해제
    array_destroy(arr);

    return 0;
}
