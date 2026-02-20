"""
스택과 큐 활용
Stack and Queue Applications

스택(LIFO)과 큐(FIFO) 자료구조를 활용한 알고리즘 예제입니다.
"""

from collections import deque
from typing import List, Optional


# =============================================================================
# 스택 구현 (리스트 기반)
# =============================================================================
class Stack:
    """리스트 기반 스택 구현"""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self.is_empty():
            return self._items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self._items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# =============================================================================
# 큐 구현 (deque 기반)
# =============================================================================
class Queue:
    """deque 기반 큐 구현 (O(1) 연산)"""

    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self._items.popleft()
        raise IndexError("Queue is empty")

    def front(self):
        if not self.is_empty():
            return self._items[0]
        raise IndexError("Queue is empty")

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# =============================================================================
# 1. 괄호 검증 (Valid Parentheses)
# =============================================================================
def is_valid_parentheses(s: str) -> bool:
    """
    괄호 문자열의 유효성 검사
    시간복잡도: O(n), 공간복잡도: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # 닫는 괄호일 때
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # 여는 괄호일 때
            stack.append(char)

    return len(stack) == 0


# =============================================================================
# 2. 후위 표기법 계산 (Postfix Evaluation)
# =============================================================================
def evaluate_postfix(expression: str) -> int:
    """
    후위 표기법 수식 계산
    예: "2 3 + 4 *" = (2 + 3) * 4 = 20
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in expression.split():
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))  # 정수 나눗셈
        else:
            stack.append(int(token))

    return stack[0]


# =============================================================================
# 3. 다음 큰 요소 (Next Greater Element)
# =============================================================================
def next_greater_element(arr: List[int]) -> List[int]:
    """
    각 요소의 오른쪽에서 처음으로 더 큰 요소 찾기
    없으면 -1 반환
    시간복잡도: O(n), 공간복잡도: O(n)
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # 인덱스 저장

    for i in range(n):
        # 현재 요소가 스택 top보다 크면
        while stack and arr[i] > arr[stack[-1]]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result


# =============================================================================
# 4. 일일 온도 (Daily Temperatures)
# =============================================================================
def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    더 따뜻한 날까지 며칠을 기다려야 하는지 계산
    시간복잡도: O(n), 공간복잡도: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # (인덱스) 저장

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result


# =============================================================================
# 5. 최소 스택 (Min Stack)
# =============================================================================
class MinStack:
    """
    O(1) 시간에 최솟값을 반환하는 스택
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1] if self.stack else None

    def get_min(self) -> int:
        return self.min_stack[-1] if self.min_stack else None


# =============================================================================
# 6. 큐 두 개로 스택 구현
# =============================================================================
class StackUsingQueues:
    """두 개의 큐로 스택 구현"""

    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x: int):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self) -> int:
        return self.q1.popleft() if self.q1 else None

    def top(self) -> int:
        return self.q1[0] if self.q1 else None

    def empty(self) -> bool:
        return len(self.q1) == 0


# =============================================================================
# 7. 슬라이딩 윈도우 최댓값 (Monotonic Queue)
# =============================================================================
def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """
    크기 k인 슬라이딩 윈도우에서 최댓값 찾기
    단조 감소 큐(Monotonic Deque) 사용
    시간복잡도: O(n), 공간복잡도: O(k)
    """
    result = []
    dq = deque()  # 인덱스 저장, 값은 단조 감소

    for i in range(len(nums)):
        # 윈도우 범위 밖의 인덱스 제거
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 현재 값보다 작은 값들 제거 (뒤에서부터)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # 윈도우가 완성되면 최댓값 추가
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("스택과 큐 활용 예제")
    print("=" * 60)

    # 1. 괄호 검증
    print("\n[1] 괄호 검증")
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    for tc in test_cases:
        result = is_valid_parentheses(tc)
        print(f"    '{tc}' -> {result}")

    # 2. 후위 표기법 계산
    print("\n[2] 후위 표기법 계산")
    expressions = ["2 3 +", "2 3 + 4 *", "5 1 2 + 4 * + 3 -"]
    for expr in expressions:
        result = evaluate_postfix(expr)
        print(f"    '{expr}' = {result}")

    # 3. 다음 큰 요소
    print("\n[3] 다음 큰 요소 (Next Greater Element)")
    arr = [4, 5, 2, 25]
    result = next_greater_element(arr)
    print(f"    배열: {arr}")
    print(f"    결과: {result}")

    # 4. 일일 온도
    print("\n[4] 일일 온도")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = daily_temperatures(temps)
    print(f"    온도: {temps}")
    print(f"    대기 일수: {result}")

    # 5. 최소 스택
    print("\n[5] 최소 스택 (MinStack)")
    min_stack = MinStack()
    operations = [
        ("push", 3), ("push", 5), ("getMin", None),
        ("push", 2), ("push", 1), ("getMin", None),
        ("pop", None), ("getMin", None)
    ]
    for op, val in operations:
        if op == "push":
            min_stack.push(val)
            print(f"    push({val})")
        elif op == "pop":
            min_stack.pop()
            print(f"    pop()")
        elif op == "getMin":
            print(f"    getMin() = {min_stack.get_min()}")

    # 6. 슬라이딩 윈도우 최댓값
    print("\n[6] 슬라이딩 윈도우 최댓값")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = max_sliding_window(nums, k)
    print(f"    배열: {nums}, k={k}")
    print(f"    각 윈도우 최댓값: {result}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
