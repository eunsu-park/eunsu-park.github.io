"""
트리와 이진 탐색 트리 (Tree & BST)
Tree and Binary Search Tree

트리 구조와 BST 연산을 구현합니다.
"""

from typing import List, Optional, Generator
from collections import deque


# =============================================================================
# 1. 이진 트리 노드
# =============================================================================

class TreeNode:
    """이진 트리 노드"""

    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


# =============================================================================
# 2. 트리 순회 (Tree Traversal)
# =============================================================================

def preorder_recursive(root: TreeNode) -> List[int]:
    """전위 순회 (재귀) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)

    traverse(root)
    return result


def preorder_iterative(root: TreeNode) -> List[int]:
    """전위 순회 (반복) - O(n)"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # 오른쪽 먼저 push (왼쪽이 먼저 처리되도록)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def inorder_recursive(root: TreeNode) -> List[int]:
    """중위 순회 (재귀) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)

    traverse(root)
    return result


def inorder_iterative(root: TreeNode) -> List[int]:
    """중위 순회 (반복) - O(n)"""
    result = []
    stack = []
    current = root

    while stack or current:
        # 왼쪽 끝까지 이동
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result


def postorder_recursive(root: TreeNode) -> List[int]:
    """후위 순회 (재귀) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node.val)

    traverse(root)
    return result


def postorder_iterative(root: TreeNode) -> List[int]:
    """후위 순회 (반복) - O(n)"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]  # 역순


def level_order(root: TreeNode) -> List[List[int]]:
    """레벨 순회 (BFS) - O(n)"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


# =============================================================================
# 3. 이진 탐색 트리 (BST)
# =============================================================================

class BST:
    """이진 탐색 트리"""

    def __init__(self):
        self.root: Optional[TreeNode] = None

    def insert(self, val: int) -> None:
        """노드 삽입 - 평균 O(log n), 최악 O(n)"""
        if not self.root:
            self.root = TreeNode(val)
            return

        current = self.root
        while True:
            if val < current.val:
                if current.left is None:
                    current.left = TreeNode(val)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(val)
                    return
                current = current.right

    def search(self, val: int) -> Optional[TreeNode]:
        """노드 검색 - 평균 O(log n), 최악 O(n)"""
        current = self.root

        while current:
            if val == current.val:
                return current
            elif val < current.val:
                current = current.left
            else:
                current = current.right

        return None

    def delete(self, val: int) -> bool:
        """노드 삭제 - 평균 O(log n), 최악 O(n)"""

        def find_min(node: TreeNode) -> TreeNode:
            while node.left:
                node = node.left
            return node

        def delete_recursive(node: TreeNode, val: int) -> Optional[TreeNode]:
            if not node:
                return None

            if val < node.val:
                node.left = delete_recursive(node.left, val)
            elif val > node.val:
                node.right = delete_recursive(node.right, val)
            else:
                # 삭제할 노드 발견

                # Case 1: 리프 노드
                if not node.left and not node.right:
                    return None

                # Case 2: 자식이 하나
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left

                # Case 3: 자식이 둘 - 후계자(오른쪽 서브트리의 최소값)로 대체
                successor = find_min(node.right)
                node.val = successor.val
                node.right = delete_recursive(node.right, successor.val)

            return node

        old_root = self.root
        self.root = delete_recursive(self.root, val)
        return old_root != self.root or (self.root and old_root.val != val if old_root else False)

    def inorder(self) -> List[int]:
        """중위 순회 (정렬된 순서)"""
        return inorder_recursive(self.root)

    def find_min(self) -> Optional[int]:
        """최솟값 찾기 - O(h)"""
        if not self.root:
            return None

        current = self.root
        while current.left:
            current = current.left
        return current.val

    def find_max(self) -> Optional[int]:
        """최댓값 찾기 - O(h)"""
        if not self.root:
            return None

        current = self.root
        while current.right:
            current = current.right
        return current.val


# =============================================================================
# 4. 트리 속성 검사
# =============================================================================

def tree_height(root: TreeNode) -> int:
    """트리 높이 계산 - O(n)"""
    if not root:
        return -1  # 빈 트리는 높이 -1, 노드 1개는 높이 0

    return 1 + max(tree_height(root.left), tree_height(root.right))


def is_balanced(root: TreeNode) -> bool:
    """균형 트리 검사 - O(n)"""

    def check(node: TreeNode) -> int:
        if not node:
            return 0

        left_height = check(node.left)
        if left_height == -1:
            return -1

        right_height = check(node.right)
        if right_height == -1:
            return -1

        if abs(left_height - right_height) > 1:
            return -1

        return 1 + max(left_height, right_height)

    return check(root) != -1


def is_valid_bst(root: TreeNode) -> bool:
    """유효한 BST 검사 - O(n)"""

    def validate(node: TreeNode, min_val: float, max_val: float) -> bool:
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))


def count_nodes(root: TreeNode) -> int:
    """노드 개수 - O(n)"""
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)


# =============================================================================
# 5. 트리 변환/구성
# =============================================================================

def build_tree_from_list(values: List[Optional[int]]) -> Optional[TreeNode]:
    """레벨 순서 리스트로 트리 구성 - O(n)"""
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def sorted_array_to_bst(nums: List[int]) -> Optional[TreeNode]:
    """정렬된 배열로 균형 BST 구성 - O(n)"""
    if not nums:
        return None

    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None

        mid = (left + right) // 2
        node = TreeNode(nums[mid])
        node.left = build(left, mid - 1)
        node.right = build(mid + 1, right)
        return node

    return build(0, len(nums) - 1)


def invert_tree(root: TreeNode) -> TreeNode:
    """트리 좌우 반전 - O(n)"""
    if not root:
        return None

    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


# =============================================================================
# 6. 실전 문제
# =============================================================================

def lowest_common_ancestor(root: TreeNode, p: int, q: int) -> Optional[TreeNode]:
    """BST에서 최소 공통 조상 (LCA) - O(h)"""
    current = root

    while current:
        if p < current.val and q < current.val:
            current = current.left
        elif p > current.val and q > current.val:
            current = current.right
        else:
            return current

    return None


def kth_smallest(root: TreeNode, k: int) -> int:
    """BST에서 k번째 작은 값 - O(h + k)"""
    stack = []
    current = root
    count = 0

    while stack or current:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        count += 1

        if count == k:
            return current.val

        current = current.right

    return -1


def path_sum(root: TreeNode, target: int) -> bool:
    """루트~리프 경로 합 확인 - O(n)"""
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == target

    remaining = target - root.val
    return path_sum(root.left, remaining) or path_sum(root.right, remaining)


def serialize(root: TreeNode) -> str:
    """트리 직렬화 - O(n)"""
    if not root:
        return "[]"

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")

    # 끝의 null 제거
    while result and result[-1] == "null":
        result.pop()

    return "[" + ",".join(result) + "]"


# =============================================================================
# 유틸리티: 트리 시각화
# =============================================================================

def print_tree(root: TreeNode, prefix: str = "", is_left: bool = True) -> None:
    """트리 ASCII 출력"""
    if not root:
        return

    print(prefix + ("├── " if is_left else "└── ") + str(root.val))

    children = []
    if root.left:
        children.append((root.left, True))
    if root.right:
        children.append((root.right, False))

    for i, (child, is_left_child) in enumerate(children):
        extension = "│   " if is_left and i < len(children) - 1 else "    "
        print_tree(child, prefix + extension, is_left_child)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("트리와 이진 탐색 트리 (Tree & BST) 예제")
    print("=" * 60)

    # 1. 트리 구성
    print("\n[1] 트리 구성")
    #       4
    #      / \
    #     2   6
    #    / \ / \
    #   1  3 5  7
    root = build_tree_from_list([4, 2, 6, 1, 3, 5, 7])
    print("    레벨 순서: [4, 2, 6, 1, 3, 5, 7]")
    print("    트리 구조:")
    print_tree(root, "    ")

    # 2. 트리 순회
    print("\n[2] 트리 순회")
    print(f"    전위 (Preorder):  {preorder_recursive(root)}")
    print(f"    중위 (Inorder):   {inorder_recursive(root)}")
    print(f"    후위 (Postorder): {postorder_recursive(root)}")
    print(f"    레벨 (Level):     {level_order(root)}")

    # 3. 트리 속성
    print("\n[3] 트리 속성")
    print(f"    높이: {tree_height(root)}")
    print(f"    노드 수: {count_nodes(root)}")
    print(f"    균형 트리: {is_balanced(root)}")
    print(f"    유효한 BST: {is_valid_bst(root)}")

    # 4. BST 연산
    print("\n[4] BST 연산")
    bst = BST()
    for val in [5, 3, 7, 1, 4, 6, 8]:
        bst.insert(val)
    print(f"    삽입: [5, 3, 7, 1, 4, 6, 8]")
    print(f"    중위 순회: {bst.inorder()}")
    print(f"    검색 4: {bst.search(4)}")
    print(f"    최솟값: {bst.find_min()}, 최댓값: {bst.find_max()}")

    bst.delete(3)
    print(f"    삭제 3 후: {bst.inorder()}")

    # 5. 정렬 배열 → 균형 BST
    print("\n[5] 정렬 배열 → 균형 BST")
    arr = [1, 2, 3, 4, 5, 6, 7]
    balanced_bst = sorted_array_to_bst(arr)
    print(f"    입력: {arr}")
    print(f"    레벨 순회: {level_order(balanced_bst)}")

    # 6. LCA
    print("\n[6] 최소 공통 조상 (LCA)")
    lca = lowest_common_ancestor(root, 1, 3)
    print(f"    노드 1, 3의 LCA: {lca.val if lca else None}")
    lca = lowest_common_ancestor(root, 1, 6)
    print(f"    노드 1, 6의 LCA: {lca.val if lca else None}")

    # 7. k번째 작은 값
    print("\n[7] k번째 작은 값")
    for k in [1, 3, 5]:
        print(f"    {k}번째 작은 값: {kth_smallest(root, k)}")

    # 8. 경로 합
    print("\n[8] 루트~리프 경로 합")
    print(f"    합 7 (4→2→1): {path_sum(root, 7)}")
    print(f"    합 10 (4→6): {path_sum(root, 10)}")

    # 9. 직렬화
    print("\n[9] 트리 직렬화")
    print(f"    {serialize(root)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
