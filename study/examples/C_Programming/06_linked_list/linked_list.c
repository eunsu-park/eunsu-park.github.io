// linked_list.c
// 단일 연결 리스트 구현

#include <stdio.h>
#include <stdlib.h>

// 노드 구조체
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// 연결 리스트 구조체
typedef struct {
    Node* head;
    int size;
} LinkedList;

// 리스트 생성
LinkedList* list_create(void) {
    LinkedList* list = malloc(sizeof(LinkedList));
    if (!list) return NULL;

    list->head = NULL;
    list->size = 0;
    return list;
}

// 맨 앞에 추가
void list_push_front(LinkedList* list, int data) {
    Node* new_node = malloc(sizeof(Node));
    if (!new_node) return;

    new_node->data = data;
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
}

// 맨 뒤에 추가
void list_push_back(LinkedList* list, int data) {
    Node* new_node = malloc(sizeof(Node));
    if (!new_node) return;

    new_node->data = data;
    new_node->next = NULL;

    if (list->head == NULL) {
        list->head = new_node;
    } else {
        Node* current = list->head;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_node;
    }
    list->size++;
}

// 맨 앞 제거
int list_pop_front(LinkedList* list, int* data) {
    if (list->head == NULL) return 0;

    Node* temp = list->head;
    *data = temp->data;
    list->head = temp->next;
    free(temp);
    list->size--;

    return 1;
}

// 특정 값 찾기
Node* list_find(LinkedList* list, int data) {
    Node* current = list->head;

    while (current != NULL) {
        if (current->data == data) {
            return current;
        }
        current = current->next;
    }

    return NULL;
}

// 특정 값 삭제
int list_remove(LinkedList* list, int data) {
    if (list->head == NULL) return 0;

    // 첫 노드가 삭제 대상인 경우
    if (list->head->data == data) {
        Node* temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->size--;
        return 1;
    }

    // 중간 또는 끝 노드 삭제
    Node* current = list->head;
    while (current->next != NULL) {
        if (current->next->data == data) {
            Node* temp = current->next;
            current->next = temp->next;
            free(temp);
            list->size--;
            return 1;
        }
        current = current->next;
    }

    return 0;  // 못 찾음
}

// 리스트 출력
void list_print(LinkedList* list) {
    Node* current = list->head;

    printf("[");
    while (current != NULL) {
        printf("%d", current->data);
        if (current->next != NULL) {
            printf(" -> ");
        }
        current = current->next;
    }
    printf("]\n");
}

// 리스트 해제
void list_destroy(LinkedList* list) {
    Node* current = list->head;

    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }

    free(list);
}

int main(void) {
    LinkedList* list = list_create();

    printf("=== 연결 리스트 테스트 ===\n\n");

    // 데이터 추가
    printf("맨 뒤에 추가: 10, 20, 30\n");
    list_push_back(list, 10);
    list_push_back(list, 20);
    list_push_back(list, 30);
    list_print(list);
    printf("크기: %d\n\n", list->size);

    // 맨 앞에 추가
    printf("맨 앞에 추가: 5\n");
    list_push_front(list, 5);
    list_print(list);
    printf("\n");

    // 값 찾기
    printf("값 20 찾기: ");
    Node* found = list_find(list, 20);
    if (found) {
        printf("찾음! (주소: %p)\n", (void*)found);
    } else {
        printf("못 찾음\n");
    }
    printf("\n");

    // 값 삭제
    printf("값 20 삭제\n");
    list_remove(list, 20);
    list_print(list);
    printf("\n");

    // 맨 앞 제거
    int data;
    printf("맨 앞 제거\n");
    list_pop_front(list, &data);
    printf("제거된 값: %d\n", data);
    list_print(list);
    printf("\n");

    // 메모리 해제
    list_destroy(list);

    return 0;
}
