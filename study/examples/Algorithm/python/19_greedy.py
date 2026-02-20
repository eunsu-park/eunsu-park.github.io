"""
탐욕 알고리즘 (Greedy Algorithm)
Greedy Algorithms

매 순간 최적의 선택을 하여 전체 최적해를 구하는 알고리즘입니다.
"""

from typing import List, Tuple
import heapq


# =============================================================================
# 1. 활동 선택 문제 (Activity Selection)
# =============================================================================

def activity_selection(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    겹치지 않는 최대 활동 수 선택
    activities: [(시작, 종료), ...]
    탐욕 전략: 종료 시간이 빠른 순서로 선택
    시간복잡도: O(n log n)
    """
    # 종료 시간으로 정렬
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = []
    last_end = 0

    for start, end in sorted_activities:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


# =============================================================================
# 2. 분할 가능 배낭 문제 (Fractional Knapsack)
# =============================================================================

def fractional_knapsack(capacity: int, items: List[Tuple[int, int]]) -> float:
    """
    분할 가능한 배낭 문제
    items: [(가치, 무게), ...]
    탐욕 전략: 단위 무게당 가치가 높은 순서로 선택
    시간복잡도: O(n log n)
    """
    # 단위 무게당 가치로 정렬 (내림차순)
    sorted_items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)

    total_value = 0
    remaining = capacity

    for value, weight in sorted_items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            # 분할해서 담기
            total_value += value * (remaining / weight)
            break

    return total_value


# =============================================================================
# 3. 회의실 배정 (Meeting Rooms)
# =============================================================================

def min_meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    필요한 최소 회의실 수
    시간복잡도: O(n log n)
    """
    if not intervals:
        return 0

    # 시작/종료 이벤트로 분리
    events = []
    for start, end in intervals:
        events.append((start, 1))   # 시작: +1
        events.append((end, -1))    # 종료: -1

    events.sort()

    max_rooms = current_rooms = 0
    for _, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms


# =============================================================================
# 4. 작업 스케줄링 (Job Scheduling)
# =============================================================================

def job_scheduling(jobs: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """
    마감 기한 내 최대 이익 스케줄링
    jobs: [(작업ID, 마감일, 이익), ...]
    탐욕 전략: 이익이 높은 순서로, 가능한 늦은 슬롯에 배정
    """
    # 이익 내림차순 정렬
    sorted_jobs = sorted(jobs, key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)
    slots = [None] * (max_deadline + 1)  # 각 시간 슬롯
    total_profit = 0
    scheduled = []

    for job_id, deadline, profit in sorted_jobs:
        # 마감일부터 역순으로 빈 슬롯 찾기
        for slot in range(deadline, 0, -1):
            if slots[slot] is None:
                slots[slot] = job_id
                total_profit += profit
                scheduled.append(job_id)
                break

    return total_profit, scheduled


# =============================================================================
# 5. 허프만 코딩 (Huffman Coding)
# =============================================================================

class HuffmanNode:
    def __init__(self, char: str = None, freq: int = 0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encoding(text: str) -> Tuple[dict, str]:
    """
    허프만 인코딩
    반환: (문자→코드 딕셔너리, 인코딩된 문자열)
    시간복잡도: O(n log n)
    """
    if not text:
        return {}, ""

    # 빈도 계산
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # 우선순위 큐로 트리 구성
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    # 코드 생성
    codes = {}

    def generate_codes(node: HuffmanNode, code: str):
        if node is None:
            return

        if node.char is not None:
            codes[node.char] = code if code else "0"
            return

        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    if heap:
        generate_codes(heap[0], "")

    # 인코딩
    encoded = ''.join(codes[char] for char in text)

    return codes, encoded


def huffman_decoding(encoded: str, codes: dict) -> str:
    """허프만 디코딩"""
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = []
    current = ""

    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""

    return ''.join(decoded)


# =============================================================================
# 6. 동전 거스름돈 (Coin Change - Greedy)
# =============================================================================

def coin_change_greedy(coins: List[int], amount: int) -> List[int]:
    """
    동전 거스름돈 (탐욕, 특정 동전 집합에서만 최적)
    coins: 내림차순 정렬된 동전
    주의: 일반적인 동전 조합에서는 DP 필요
    """
    coins = sorted(coins, reverse=True)
    result = []

    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin

    return result if amount == 0 else []


# =============================================================================
# 7. 구간 스케줄링 (Interval Scheduling)
# =============================================================================

def interval_partitioning(intervals: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    구간들을 최소 그룹 수로 분할 (각 그룹 내 겹침 없음)
    시간복잡도: O(n log n)
    """
    if not intervals:
        return []

    # 시작 시간으로 정렬
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: x[1][0])

    # 각 그룹의 종료 시간을 힙으로 관리
    groups = []  # [(종료시간, 그룹인덱스)]
    assignment = [[] for _ in range(len(intervals))]

    for idx, (start, end) in sorted_intervals:
        if groups and groups[0][0] <= start:
            # 기존 그룹에 배정
            _, group_idx = heapq.heappop(groups)
            assignment[group_idx].append(intervals[idx])
            heapq.heappush(groups, (end, group_idx))
        else:
            # 새 그룹 생성
            new_group = len(groups)
            assignment.append([intervals[idx]])
            heapq.heappush(groups, (end, new_group))

    return [g for g in assignment if g]


# =============================================================================
# 8. 점프 게임 (Jump Game)
# =============================================================================

def can_jump(nums: List[int]) -> bool:
    """
    마지막 인덱스 도달 가능 여부
    nums[i] = i에서 최대 점프 거리
    시간복잡도: O(n)
    """
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)

    return True


def min_jumps(nums: List[int]) -> int:
    """
    마지막 인덱스까지 최소 점프 횟수
    시간복잡도: O(n)
    """
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= len(nums) - 1:
                break

    return jumps


# =============================================================================
# 9. 주유소 (Gas Station)
# =============================================================================

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    원형 경로 완주 가능한 시작점 찾기
    gas[i] = i에서 얻는 기름
    cost[i] = i→i+1 이동 비용
    시간복잡도: O(n)
    """
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        gain = gas[i] - cost[i]
        total_tank += gain
        current_tank += gain

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1


# =============================================================================
# 10. 최소 화살로 풍선 터트리기
# =============================================================================

def min_arrows(points: List[List[int]]) -> int:
    """
    수평선 풍선을 터트리는 최소 화살 수
    points[i] = [시작, 끝] 범위
    시간복잡도: O(n log n)
    """
    if not points:
        return 0

    # 끝점으로 정렬
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("탐욕 알고리즘 (Greedy) 예제")
    print("=" * 60)

    # 1. 활동 선택
    print("\n[1] 활동 선택 문제")
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11)]
    selected = activity_selection(activities)
    print(f"    활동: {activities}")
    print(f"    선택: {selected} ({len(selected)}개)")

    # 2. 분할 가능 배낭
    print("\n[2] 분할 가능 배낭")
    items = [(60, 10), (100, 20), (120, 30)]  # (가치, 무게)
    capacity = 50
    value = fractional_knapsack(capacity, items)
    print(f"    아이템 (가치, 무게): {items}")
    print(f"    용량: {capacity}, 최대 가치: {value}")

    # 3. 회의실 배정
    print("\n[3] 최소 회의실 수")
    meetings = [(0, 30), (5, 10), (15, 20)]
    rooms = min_meeting_rooms(meetings)
    print(f"    회의: {meetings}")
    print(f"    필요 회의실: {rooms}개")

    # 4. 작업 스케줄링
    print("\n[4] 작업 스케줄링")
    jobs = [(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)]  # (ID, 마감일, 이익)
    profit, scheduled = job_scheduling(jobs)
    print(f"    작업 (ID, 마감, 이익): {jobs}")
    print(f"    스케줄: {scheduled}, 총 이익: {profit}")

    # 5. 허프만 코딩
    print("\n[5] 허프만 코딩")
    text = "abracadabra"
    codes, encoded = huffman_encoding(text)
    decoded = huffman_decoding(encoded, codes)
    print(f"    원본: '{text}'")
    print(f"    코드: {codes}")
    print(f"    인코딩: {encoded} ({len(encoded)} bits)")
    print(f"    원본 크기: {len(text) * 8} bits")
    print(f"    디코딩: '{decoded}'")

    # 6. 동전 거스름돈
    print("\n[6] 동전 거스름돈")
    coins = [500, 100, 50, 10]
    amount = 1260
    result = coin_change_greedy(coins, amount)
    print(f"    동전: {coins}, 금액: {amount}")
    print(f"    결과: {result} ({len(result)}개)")

    # 7. 점프 게임
    print("\n[7] 점프 게임")
    nums1 = [2, 3, 1, 1, 4]
    nums2 = [3, 2, 1, 0, 4]
    print(f"    {nums1}: 도달 가능 = {can_jump(nums1)}, 최소 점프 = {min_jumps(nums1)}")
    print(f"    {nums2}: 도달 가능 = {can_jump(nums2)}")

    # 8. 주유소
    print("\n[8] 주유소 문제")
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    start = can_complete_circuit(gas, cost)
    print(f"    gas: {gas}, cost: {cost}")
    print(f"    시작점: {start}")

    # 9. 풍선 터트리기
    print("\n[9] 최소 화살로 풍선 터트리기")
    points = [[10, 16], [2, 8], [1, 6], [7, 12]]
    arrows = min_arrows(points)
    print(f"    풍선: {points}")
    print(f"    필요 화살: {arrows}개")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
