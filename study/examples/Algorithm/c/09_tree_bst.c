/*
 * 트리와 이진 탐색 트리 (Tree and BST)
 * Tree Traversal, BST Operations, Tree Properties
 *
 * 트리 자료구조의 기본 연산들입니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. 이진 트리 노드
 * ============================================================================= */

typedef struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

TreeNode* create_node(int val) {
    TreeNode* node = malloc(sizeof(TreeNode));
    node->val = val;
    node->left = NULL;
    node->right = NULL;
    return node;
}

void free_tree(TreeNode* root) {
    if (root == NULL) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

/* =============================================================================
 * 2. 트리 순회
 * ============================================================================= */

void preorder(TreeNode* root) {
    if (root == NULL) return;
    printf("%d ", root->val);
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (root == NULL) return;
    inorder(root->left);
    printf("%d ", root->val);
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->val);
}

/* 레벨 순회 (BFS) */
void level_order(TreeNode* root) {
    if (root == NULL) return;

    TreeNode** queue = malloc(1000 * sizeof(TreeNode*));
    int front = 0, rear = 0;
    queue[rear++] = root;

    while (front < rear) {
        TreeNode* node = queue[front++];
        printf("%d ", node->val);

        if (node->left) queue[rear++] = node->left;
        if (node->right) queue[rear++] = node->right;
    }

    free(queue);
}

/* =============================================================================
 * 3. BST 연산
 * ============================================================================= */

TreeNode* bst_insert(TreeNode* root, int val) {
    if (root == NULL) return create_node(val);

    if (val < root->val)
        root->left = bst_insert(root->left, val);
    else if (val > root->val)
        root->right = bst_insert(root->right, val);

    return root;
}

TreeNode* bst_search(TreeNode* root, int val) {
    if (root == NULL || root->val == val)
        return root;

    if (val < root->val)
        return bst_search(root->left, val);
    return bst_search(root->right, val);
}

TreeNode* find_min(TreeNode* root) {
    while (root && root->left)
        root = root->left;
    return root;
}

TreeNode* find_max(TreeNode* root) {
    while (root && root->right)
        root = root->right;
    return root;
}

TreeNode* bst_delete(TreeNode* root, int val) {
    if (root == NULL) return NULL;

    if (val < root->val) {
        root->left = bst_delete(root->left, val);
    } else if (val > root->val) {
        root->right = bst_delete(root->right, val);
    } else {
        /* 노드 찾음 */
        if (root->left == NULL) {
            TreeNode* temp = root->right;
            free(root);
            return temp;
        } else if (root->right == NULL) {
            TreeNode* temp = root->left;
            free(root);
            return temp;
        }

        /* 두 자식 있음: 오른쪽 서브트리의 최솟값으로 대체 */
        TreeNode* successor = find_min(root->right);
        root->val = successor->val;
        root->right = bst_delete(root->right, successor->val);
    }

    return root;
}

/* =============================================================================
 * 4. 트리 속성
 * ============================================================================= */

int tree_height(TreeNode* root) {
    if (root == NULL) return 0;

    int left_h = tree_height(root->left);
    int right_h = tree_height(root->right);

    return 1 + (left_h > right_h ? left_h : right_h);
}

int count_nodes(TreeNode* root) {
    if (root == NULL) return 0;
    return 1 + count_nodes(root->left) + count_nodes(root->right);
}

int count_leaves(TreeNode* root) {
    if (root == NULL) return 0;
    if (root->left == NULL && root->right == NULL) return 1;
    return count_leaves(root->left) + count_leaves(root->right);
}

bool is_balanced(TreeNode* root) {
    if (root == NULL) return true;

    int left_h = tree_height(root->left);
    int right_h = tree_height(root->right);

    if (abs(left_h - right_h) > 1) return false;

    return is_balanced(root->left) && is_balanced(root->right);
}

/* BST 검증 */
bool is_bst_util(TreeNode* root, TreeNode* min_node, TreeNode* max_node) {
    if (root == NULL) return true;

    if (min_node && root->val <= min_node->val) return false;
    if (max_node && root->val >= max_node->val) return false;

    return is_bst_util(root->left, min_node, root) &&
           is_bst_util(root->right, root, max_node);
}

bool is_bst(TreeNode* root) {
    return is_bst_util(root, NULL, NULL);
}

/* =============================================================================
 * 5. 경로 문제
 * ============================================================================= */

/* 루트에서 리프까지의 경로 합 */
bool has_path_sum(TreeNode* root, int target_sum) {
    if (root == NULL) return false;

    if (root->left == NULL && root->right == NULL)
        return target_sum == root->val;

    return has_path_sum(root->left, target_sum - root->val) ||
           has_path_sum(root->right, target_sum - root->val);
}

/* 최대 경로 합 */
int max_path_sum_util(TreeNode* root, int* max_sum) {
    if (root == NULL) return 0;

    int left = max_path_sum_util(root->left, max_sum);
    int right = max_path_sum_util(root->right, max_sum);

    left = left > 0 ? left : 0;
    right = right > 0 ? right : 0;

    int path_sum = root->val + left + right;
    if (path_sum > *max_sum) *max_sum = path_sum;

    return root->val + (left > right ? left : right);
}

int max_path_sum(TreeNode* root) {
    int max_sum = root ? root->val : 0;
    max_path_sum_util(root, &max_sum);
    return max_sum;
}

/* =============================================================================
 * 6. LCA (Lowest Common Ancestor)
 * ============================================================================= */

/* 일반 트리 LCA */
TreeNode* lca(TreeNode* root, int p, int q) {
    if (root == NULL) return NULL;
    if (root->val == p || root->val == q) return root;

    TreeNode* left = lca(root->left, p, q);
    TreeNode* right = lca(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

/* BST LCA */
TreeNode* lca_bst(TreeNode* root, int p, int q) {
    if (root == NULL) return NULL;

    if (p < root->val && q < root->val)
        return lca_bst(root->left, p, q);
    if (p > root->val && q > root->val)
        return lca_bst(root->right, p, q);

    return root;
}

/* =============================================================================
 * 7. 트리 변환
 * ============================================================================= */

/* 좌우 반전 */
TreeNode* invert_tree(TreeNode* root) {
    if (root == NULL) return NULL;

    TreeNode* temp = root->left;
    root->left = invert_tree(root->right);
    root->right = invert_tree(temp);

    return root;
}

/* 트리 직렬화 (전위 순회) */
void serialize(TreeNode* root, int* arr, int* idx) {
    if (root == NULL) {
        arr[(*idx)++] = -1;  /* NULL 표시 */
        return;
    }
    arr[(*idx)++] = root->val;
    serialize(root->left, arr, idx);
    serialize(root->right, arr, idx);
}

TreeNode* deserialize(int* arr, int* idx, int n) {
    if (*idx >= n || arr[*idx] == -1) {
        (*idx)++;
        return NULL;
    }

    TreeNode* node = create_node(arr[(*idx)++]);
    node->left = deserialize(arr, idx, n);
    node->right = deserialize(arr, idx, n);
    return node;
}

/* =============================================================================
 * 테스트
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("트리와 BST (Tree and BST) 예제\n");
    printf("============================================================\n");

    /* BST 생성 */
    /*
     *        50
     *       /  \
     *      30   70
     *     / \   / \
     *    20 40 60 80
     */
    TreeNode* root = NULL;
    int values[] = {50, 30, 70, 20, 40, 60, 80};
    for (int i = 0; i < 7; i++) {
        root = bst_insert(root, values[i]);
    }

    /* 1. 순회 */
    printf("\n[1] 트리 순회\n");
    printf("    전위: ");
    preorder(root);
    printf("\n");
    printf("    중위: ");
    inorder(root);
    printf("\n");
    printf("    후위: ");
    postorder(root);
    printf("\n");
    printf("    레벨: ");
    level_order(root);
    printf("\n");

    /* 2. BST 연산 */
    printf("\n[2] BST 연산\n");
    printf("    검색 40: %s\n", bst_search(root, 40) ? "found" : "not found");
    printf("    검색 45: %s\n", bst_search(root, 45) ? "found" : "not found");
    printf("    최솟값: %d\n", find_min(root)->val);
    printf("    최댓값: %d\n", find_max(root)->val);

    /* 3. 트리 속성 */
    printf("\n[3] 트리 속성\n");
    printf("    높이: %d\n", tree_height(root));
    printf("    노드 수: %d\n", count_nodes(root));
    printf("    리프 수: %d\n", count_leaves(root));
    printf("    균형: %s\n", is_balanced(root) ? "yes" : "no");
    printf("    BST: %s\n", is_bst(root) ? "yes" : "no");

    /* 4. 경로 */
    printf("\n[4] 경로 문제\n");
    printf("    경로 합 100 존재: %s\n", has_path_sum(root, 100) ? "yes" : "no");
    printf("    최대 경로 합: %d\n", max_path_sum(root));

    /* 5. LCA */
    printf("\n[5] LCA (최소 공통 조상)\n");
    TreeNode* ancestor = lca_bst(root, 20, 40);
    printf("    LCA(20, 40): %d\n", ancestor ? ancestor->val : -1);
    ancestor = lca_bst(root, 20, 80);
    printf("    LCA(20, 80): %d\n", ancestor ? ancestor->val : -1);

    /* 6. 삭제 */
    printf("\n[6] BST 삭제\n");
    printf("    삭제 전 중위: ");
    inorder(root);
    printf("\n");
    root = bst_delete(root, 30);
    printf("    30 삭제 후 중위: ");
    inorder(root);
    printf("\n");

    /* 7. 직렬화 */
    printf("\n[7] 트리 직렬화/역직렬화\n");
    int serialized[100];
    int idx = 0;
    serialize(root, serialized, &idx);
    printf("    직렬화: ");
    for (int i = 0; i < idx; i++) {
        printf("%d ", serialized[i]);
    }
    printf("\n");

    int deserialize_idx = 0;
    TreeNode* restored = deserialize(serialized, &deserialize_idx, idx);
    printf("    복원 후 중위: ");
    inorder(restored);
    printf("\n");

    /* 정리 */
    free_tree(root);
    free_tree(restored);

    printf("\n============================================================\n");

    return 0;
}
