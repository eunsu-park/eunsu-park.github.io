/*
 * 위상 정렬 (Topological Sort)
 * Kahn's Algorithm, DFS-based, Cycle Detection
 *
 * 방향 비순환 그래프(DAG)의 선형 순서를 찾습니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. 그래프 구조
 * ============================================================================= */

typedef struct AdjNode {
    int vertex;
    struct AdjNode* next;
} AdjNode;

typedef struct {
    AdjNode** adj;
    int vertices;
} Graph;

Graph* graph_create(int vertices) {
    Graph* g = malloc(sizeof(Graph));
    g->vertices = vertices;
    g->adj = calloc(vertices, sizeof(AdjNode*));
    return g;
}

void graph_add_edge(Graph* g, int src, int dest) {
    AdjNode* node = malloc(sizeof(AdjNode));
    node->vertex = dest;
    node->next = g->adj[src];
    g->adj[src] = node;
}

void graph_free(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        AdjNode* node = g->adj[i];
        while (node) {
            AdjNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(g->adj);
    free(g);
}

/* =============================================================================
 * 2. Kahn 알고리즘 (BFS 기반)
 * ============================================================================= */

int* kahn_topological_sort(Graph* g, bool* has_cycle) {
    int* in_degree = calloc(g->vertices, sizeof(int));
    int* result = malloc(g->vertices * sizeof(int));
    int result_idx = 0;

    /* 진입 차수 계산 */
    for (int v = 0; v < g->vertices; v++) {
        AdjNode* node = g->adj[v];
        while (node) {
            in_degree[node->vertex]++;
            node = node->next;
        }
    }

    /* 진입 차수 0인 정점 큐에 추가 */
    int* queue = malloc(g->vertices * sizeof(int));
    int front = 0, rear = 0;

    for (int i = 0; i < g->vertices; i++) {
        if (in_degree[i] == 0)
            queue[rear++] = i;
    }

    /* BFS */
    while (front < rear) {
        int v = queue[front++];
        result[result_idx++] = v;

        AdjNode* node = g->adj[v];
        while (node) {
            in_degree[node->vertex]--;
            if (in_degree[node->vertex] == 0)
                queue[rear++] = node->vertex;
            node = node->next;
        }
    }

    *has_cycle = (result_idx != g->vertices);

    free(in_degree);
    free(queue);
    return result;
}

/* =============================================================================
 * 3. DFS 기반 위상 정렬
 * ============================================================================= */

void dfs_topological_util(Graph* g, int v, bool visited[], int stack[], int* stack_top) {
    visited[v] = true;

    AdjNode* node = g->adj[v];
    while (node) {
        if (!visited[node->vertex])
            dfs_topological_util(g, node->vertex, visited, stack, stack_top);
        node = node->next;
    }

    stack[(*stack_top)++] = v;
}

int* dfs_topological_sort(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* stack = malloc(g->vertices * sizeof(int));
    int stack_top = 0;

    for (int i = 0; i < g->vertices; i++) {
        if (!visited[i])
            dfs_topological_util(g, i, visited, stack, &stack_top);
    }

    /* 스택 역순으로 결과 생성 */
    int* result = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        result[i] = stack[g->vertices - 1 - i];

    free(visited);
    free(stack);
    return result;
}

/* =============================================================================
 * 4. 사이클 탐지를 포함한 DFS 위상 정렬
 * ============================================================================= */

typedef enum { WHITE, GRAY, BLACK } Color;

bool dfs_topological_cycle_util(Graph* g, int v, Color color[], int stack[], int* stack_top) {
    color[v] = GRAY;

    AdjNode* node = g->adj[v];
    while (node) {
        if (color[node->vertex] == GRAY)
            return true;  /* 사이클 발견 */
        if (color[node->vertex] == WHITE) {
            if (dfs_topological_cycle_util(g, node->vertex, color, stack, stack_top))
                return true;
        }
        node = node->next;
    }

    color[v] = BLACK;
    stack[(*stack_top)++] = v;
    return false;
}

int* dfs_topological_with_cycle(Graph* g, bool* has_cycle) {
    Color* color = calloc(g->vertices, sizeof(Color));
    int* stack = malloc(g->vertices * sizeof(int));
    int stack_top = 0;

    *has_cycle = false;
    for (int i = 0; i < g->vertices; i++) {
        if (color[i] == WHITE) {
            if (dfs_topological_cycle_util(g, i, color, stack, &stack_top)) {
                *has_cycle = true;
                free(color);
                free(stack);
                return NULL;
            }
        }
    }

    int* result = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        result[i] = stack[g->vertices - 1 - i];

    free(color);
    free(stack);
    return result;
}

/* =============================================================================
 * 5. 모든 위상 정렬 찾기
 * ============================================================================= */

void all_topological_sorts_util(Graph* g, bool visited[], int in_degree[],
                                 int result[], int idx, int* count) {
    bool found = false;

    for (int v = 0; v < g->vertices; v++) {
        if (!visited[v] && in_degree[v] == 0) {
            visited[v] = true;
            result[idx] = v;

            /* 인접 정점의 진입 차수 감소 */
            AdjNode* node = g->adj[v];
            while (node) {
                in_degree[node->vertex]--;
                node = node->next;
            }

            all_topological_sorts_util(g, visited, in_degree, result, idx + 1, count);

            /* 복원 */
            visited[v] = false;
            node = g->adj[v];
            while (node) {
                in_degree[node->vertex]++;
                node = node->next;
            }

            found = true;
        }
    }

    if (!found && idx == g->vertices) {
        (*count)++;
        printf("      ");
        for (int i = 0; i < g->vertices; i++)
            printf("%d ", result[i]);
        printf("\n");
    }
}

int all_topological_sorts(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* in_degree = calloc(g->vertices, sizeof(int));
    int* result = malloc(g->vertices * sizeof(int));
    int count = 0;

    /* 진입 차수 계산 */
    for (int v = 0; v < g->vertices; v++) {
        AdjNode* node = g->adj[v];
        while (node) {
            in_degree[node->vertex]++;
            node = node->next;
        }
    }

    all_topological_sorts_util(g, visited, in_degree, result, 0, &count);

    free(visited);
    free(in_degree);
    free(result);
    return count;
}

/* =============================================================================
 * 6. 실전: 과목 이수 순서
 * ============================================================================= */

bool can_finish_courses(int num_courses, int prerequisites[][2], int num_prereqs) {
    Graph* g = graph_create(num_courses);

    for (int i = 0; i < num_prereqs; i++) {
        graph_add_edge(g, prerequisites[i][1], prerequisites[i][0]);
    }

    bool has_cycle;
    int* result = kahn_topological_sort(g, &has_cycle);

    free(result);
    graph_free(g);

    return !has_cycle;
}

int* find_course_order(int num_courses, int prerequisites[][2], int num_prereqs,
                       int* order_size) {
    Graph* g = graph_create(num_courses);

    for (int i = 0; i < num_prereqs; i++) {
        graph_add_edge(g, prerequisites[i][1], prerequisites[i][0]);
    }

    bool has_cycle;
    int* result = kahn_topological_sort(g, &has_cycle);

    graph_free(g);

    if (has_cycle) {
        free(result);
        *order_size = 0;
        return NULL;
    }

    *order_size = num_courses;
    return result;
}

/* =============================================================================
 * 7. 실전: 작업 스케줄링
 * ============================================================================= */

int* task_scheduling(int num_tasks, int dependencies[][2], int num_deps,
                     int task_times[], int* completion_times) {
    Graph* g = graph_create(num_tasks);

    for (int i = 0; i < num_deps; i++) {
        graph_add_edge(g, dependencies[i][0], dependencies[i][1]);
    }

    bool has_cycle;
    int* order = kahn_topological_sort(g, &has_cycle);

    if (has_cycle) {
        graph_free(g);
        free(order);
        return NULL;
    }

    /* 최단 완료 시간 계산 */
    for (int i = 0; i < num_tasks; i++)
        completion_times[i] = task_times[i];

    for (int i = 0; i < num_tasks; i++) {
        int v = order[i];
        AdjNode* node = g->adj[v];
        while (node) {
            int new_time = completion_times[v] + task_times[node->vertex];
            if (new_time > completion_times[node->vertex])
                completion_times[node->vertex] = new_time;
            node = node->next;
        }
    }

    graph_free(g);
    return order;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

void print_array(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
}

int main(void) {
    printf("============================================================\n");
    printf("위상 정렬 (Topological Sort) 예제\n");
    printf("============================================================\n");

    /* 1. Kahn 알고리즘 */
    printf("\n[1] Kahn 알고리즘 (BFS)\n");
    Graph* g1 = graph_create(6);
    graph_add_edge(g1, 5, 2);
    graph_add_edge(g1, 5, 0);
    graph_add_edge(g1, 4, 0);
    graph_add_edge(g1, 4, 1);
    graph_add_edge(g1, 2, 3);
    graph_add_edge(g1, 3, 1);

    bool has_cycle;
    int* result = kahn_topological_sort(g1, &has_cycle);
    printf("    그래프: 5→2, 5→0, 4→0, 4→1, 2→3, 3→1\n");
    printf("    위상 정렬: ");
    if (!has_cycle) print_array(result, 6);
    printf("\n");
    free(result);

    /* 2. DFS 기반 */
    printf("\n[2] DFS 기반 위상 정렬\n");
    result = dfs_topological_sort(g1);
    printf("    위상 정렬: ");
    print_array(result, 6);
    printf("\n");
    free(result);

    /* 3. 사이클 탐지 */
    printf("\n[3] 사이클 탐지\n");
    Graph* g2 = graph_create(3);
    graph_add_edge(g2, 0, 1);
    graph_add_edge(g2, 1, 2);
    graph_add_edge(g2, 2, 0);  /* 사이클 */

    result = dfs_topological_with_cycle(g2, &has_cycle);
    printf("    그래프 0→1→2→0 (사이클): %s\n",
           has_cycle ? "사이클 발견" : "DAG");
    if (result) free(result);
    graph_free(g2);

    /* 4. 모든 위상 정렬 */
    printf("\n[4] 모든 위상 정렬\n");
    Graph* g3 = graph_create(4);
    graph_add_edge(g3, 0, 1);
    graph_add_edge(g3, 0, 2);
    graph_add_edge(g3, 1, 3);
    graph_add_edge(g3, 2, 3);

    printf("    그래프: 0→1, 0→2, 1→3, 2→3\n");
    printf("    모든 순서:\n");
    int count = all_topological_sorts(g3);
    printf("    총 %d개\n", count);
    graph_free(g3);

    /* 5. 과목 이수 */
    printf("\n[5] 과목 이수 순서\n");
    int prereqs[][2] = {{1, 0}, {2, 0}, {3, 1}, {3, 2}};
    int order_size;
    int* course_order = find_course_order(4, prereqs, 4, &order_size);

    printf("    과목: 0, 1, 2, 3\n");
    printf("    선수: 1←0, 2←0, 3←1, 3←2\n");
    if (course_order) {
        printf("    이수 순서: ");
        print_array(course_order, order_size);
        printf("\n");
        free(course_order);
    }

    /* 6. 작업 스케줄링 */
    printf("\n[6] 작업 스케줄링\n");
    int deps[][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    int times[] = {2, 3, 4, 1};  /* 각 작업 소요 시간 */
    int completion[4];

    int* task_order = task_scheduling(4, deps, 4, times, completion);
    printf("    작업 시간: [2, 3, 4, 1]\n");
    printf("    의존성: 0→1, 0→2, 1→3, 2→3\n");
    if (task_order) {
        printf("    완료 시간: ");
        for (int i = 0; i < 4; i++)
            printf("%d ", completion[i]);
        printf("\n");
        free(task_order);
    }

    graph_free(g1);

    /* 7. 알고리즘 비교 */
    printf("\n[7] 알고리즘 비교\n");
    printf("    | 방법      | 시간복잡도 | 특징                  |\n");
    printf("    |-----------|------------|------------------------|\n");
    printf("    | Kahn(BFS) | O(V + E)   | 사이클 탐지 용이      |\n");
    printf("    | DFS       | O(V + E)   | 구현 간단             |\n");
    printf("    | 모든 정렬 | O(V! * E)  | 모든 경우 열거        |\n");

    printf("\n============================================================\n");

    return 0;
}
