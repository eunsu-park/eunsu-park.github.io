/*
 * 트리와 이진 탐색 트리 (Tree and BST)
 * Tree Traversal, BST Operations, LCA
 *
 * 계층적 자료구조의 구현과 활용입니다.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. 이진 트리 노드
// =============================================================================

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// =============================================================================
// 2. 트리 순회
// =============================================================================

// 전위 순회 (Preorder)
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// 중위 순회 (Inorder)
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// 후위 순회 (Postorder)
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);
}

// 레벨 순회 (Level Order)
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }

    return result;
}

// 반복적 중위 순회
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;

    while (curr || !st.empty()) {
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }

    return result;
}

// =============================================================================
// 3. BST 연산
// =============================================================================

class BST {
public:
    TreeNode* root;

    BST() : root(nullptr) {}

    // 삽입
    TreeNode* insert(TreeNode* node, int val) {
        if (!node) return new TreeNode(val);

        if (val < node->val)
            node->left = insert(node->left, val);
        else if (val > node->val)
            node->right = insert(node->right, val);

        return node;
    }

    void insert(int val) {
        root = insert(root, val);
    }

    // 검색
    TreeNode* search(TreeNode* node, int val) {
        if (!node || node->val == val) return node;

        if (val < node->val)
            return search(node->left, val);
        return search(node->right, val);
    }

    bool search(int val) {
        return search(root, val) != nullptr;
    }

    // 최솟값
    TreeNode* findMin(TreeNode* node) {
        while (node && node->left)
            node = node->left;
        return node;
    }

    // 삭제
    TreeNode* remove(TreeNode* node, int val) {
        if (!node) return nullptr;

        if (val < node->val) {
            node->left = remove(node->left, val);
        } else if (val > node->val) {
            node->right = remove(node->right, val);
        } else {
            // 자식이 없거나 하나
            if (!node->left) {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            }
            if (!node->right) {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }

            // 자식이 둘
            TreeNode* successor = findMin(node->right);
            node->val = successor->val;
            node->right = remove(node->right, successor->val);
        }
        return node;
    }

    void remove(int val) {
        root = remove(root, val);
    }
};

// =============================================================================
// 4. 트리 속성
// =============================================================================

// 높이
int height(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(height(root->left), height(root->right));
}

// 노드 개수
int countNodes(TreeNode* root) {
    if (!root) return 0;
    return 1 + countNodes(root->left) + countNodes(root->right);
}

// 균형 검사
bool isBalanced(TreeNode* root) {
    if (!root) return true;

    int leftH = height(root->left);
    int rightH = height(root->right);

    return abs(leftH - rightH) <= 1 &&
           isBalanced(root->left) &&
           isBalanced(root->right);
}

// BST 유효성 검사
bool isValidBST(TreeNode* root, long long minVal = LLONG_MIN, long long maxVal = LLONG_MAX) {
    if (!root) return true;

    if (root->val <= minVal || root->val >= maxVal)
        return false;

    return isValidBST(root->left, minVal, root->val) &&
           isValidBST(root->right, root->val, maxVal);
}

// =============================================================================
// 5. LCA (Lowest Common Ancestor)
// =============================================================================

// 일반 이진 트리의 LCA
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

// BST의 LCA
TreeNode* lcaBST(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val)
            root = root->left;
        else if (p->val > root->val && q->val > root->val)
            root = root->right;
        else
            return root;
    }
    return nullptr;
}

// =============================================================================
// 6. 경로 합
// =============================================================================

// 루트에서 리프까지 합이 target인 경로 존재 여부
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;

    if (!root->left && !root->right)
        return targetSum == root->val;

    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
}

// 모든 경로 찾기
void pathSumHelper(TreeNode* root, int sum, vector<int>& path, vector<vector<int>>& result) {
    if (!root) return;

    path.push_back(root->val);

    if (!root->left && !root->right && sum == root->val) {
        result.push_back(path);
    } else {
        pathSumHelper(root->left, sum - root->val, path, result);
        pathSumHelper(root->right, sum - root->val, path, result);
    }

    path.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    pathSumHelper(root, targetSum, path, result);
    return result;
}

// =============================================================================
// 테스트
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "트리와 BST 예제" << endl;
    cout << "============================================================" << endl;

    // 테스트 트리 생성
    //       4
    //      / \
    //     2   6
    //    / \ / \
    //   1  3 5  7
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(5);
    root->right->right = new TreeNode(7);

    // 1. 트리 순회
    cout << "\n[1] 트리 순회" << endl;
    vector<int> pre, in, post;
    preorder(root, pre);
    inorder(root, in);
    postorder(root, post);

    cout << "    전위: ";
    printVector(pre);
    cout << endl;
    cout << "    중위: ";
    printVector(in);
    cout << endl;
    cout << "    후위: ";
    printVector(post);
    cout << endl;

    // 2. 레벨 순회
    cout << "\n[2] 레벨 순회" << endl;
    auto levels = levelOrder(root);
    for (size_t i = 0; i < levels.size(); i++) {
        cout << "    레벨 " << i << ": ";
        printVector(levels[i]);
        cout << endl;
    }

    // 3. BST 연산
    cout << "\n[3] BST 연산" << endl;
    BST bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);

    cout << "    삽입: 50, 30, 70, 20, 40" << endl;
    cout << "    검색 30: " << (bst.search(30) ? "있음" : "없음") << endl;
    cout << "    검색 60: " << (bst.search(60) ? "있음" : "없음") << endl;

    // 4. 트리 속성
    cout << "\n[4] 트리 속성" << endl;
    cout << "    높이: " << height(root) << endl;
    cout << "    노드 수: " << countNodes(root) << endl;
    cout << "    균형 여부: " << (isBalanced(root) ? "예" : "아니오") << endl;
    cout << "    유효한 BST: " << (isValidBST(root) ? "예" : "아니오") << endl;

    // 5. LCA
    cout << "\n[5] 최소 공통 조상 (LCA)" << endl;
    TreeNode* lca = lowestCommonAncestor(root, root->left->left, root->left->right);
    cout << "    LCA(1, 3) = " << lca->val << endl;
    lca = lowestCommonAncestor(root, root->left, root->right);
    cout << "    LCA(2, 6) = " << lca->val << endl;

    // 6. 경로 합
    cout << "\n[6] 경로 합" << endl;
    cout << "    합 7 경로 존재: " << (hasPathSum(root, 7) ? "예" : "아니오") << endl;
    cout << "    합 15 경로 존재: " << (hasPathSum(root, 15) ? "예" : "아니오") << endl;

    // 7. 복잡도 요약
    cout << "\n[7] 복잡도 요약" << endl;
    cout << "    | 연산       | 평균      | 최악      |" << endl;
    cout << "    |------------|-----------|-----------|" << endl;
    cout << "    | 검색       | O(log n)  | O(n)      |" << endl;
    cout << "    | 삽입       | O(log n)  | O(n)      |" << endl;
    cout << "    | 삭제       | O(log n)  | O(n)      |" << endl;
    cout << "    | 순회       | O(n)      | O(n)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
