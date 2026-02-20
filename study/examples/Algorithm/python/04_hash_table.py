"""
해시 테이블 (Hash Table)
Hash Table Implementation

해시 함수와 충돌 해결 방법을 구현합니다.
"""

from typing import Any, List, Optional, Tuple
from collections import OrderedDict


# =============================================================================
# 1. 해시 함수 (Hash Functions)
# =============================================================================

def hash_division(key: int, table_size: int) -> int:
    """나눗셈 해시 함수 - O(1)"""
    return key % table_size


def hash_multiplication(key: int, table_size: int, A: float = 0.6180339887) -> int:
    """곱셈 해시 함수 (Knuth 제안 A값) - O(1)"""
    return int(table_size * ((key * A) % 1))


def hash_string(s: str, table_size: int) -> int:
    """문자열 해시 함수 (다항식 롤링 해시) - O(n)"""
    hash_val = 0
    base = 31
    for char in s:
        hash_val = (hash_val * base + ord(char)) % table_size
    return hash_val


# =============================================================================
# 2. 체이닝 (Chaining)
# =============================================================================

class HashTableChaining:
    """체이닝 방식 해시 테이블"""

    def __init__(self, size: int = 10):
        self.size = size
        self.table: List[List[Tuple[Any, Any]]] = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key: Any) -> int:
        """해시 함수"""
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def put(self, key: Any, value: Any) -> None:
        """키-값 삽입 - 평균 O(1), 최악 O(n)"""
        index = self._hash(key)

        # 기존 키 업데이트
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return

        # 새 키 추가
        self.table[index].append((key, value))
        self.count += 1

    def get(self, key: Any) -> Optional[Any]:
        """키로 값 조회 - 평균 O(1), 최악 O(n)"""
        index = self._hash(key)

        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def remove(self, key: Any) -> bool:
        """키-값 삭제 - 평균 O(1), 최악 O(n)"""
        index = self._hash(key)

        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                return True
        return False

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def load_factor(self) -> float:
        return self.count / self.size


# =============================================================================
# 3. 오픈 어드레싱 - 선형 탐사 (Linear Probing)
# =============================================================================

class HashTableLinearProbing:
    """선형 탐사 방식 해시 테이블"""

    DELETED = object()  # 삭제 마커

    def __init__(self, size: int = 10):
        self.size = size
        self.keys: List[Any] = [None] * size
        self.values: List[Any] = [None] * size
        self.count = 0

    def _hash(self, key: Any) -> int:
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def _probe(self, key: Any, for_insert: bool = False) -> int:
        """선형 탐사로 슬롯 찾기"""
        index = self._hash(key)
        first_deleted = -1

        for i in range(self.size):
            probe_index = (index + i) % self.size

            if self.keys[probe_index] is None:
                if for_insert and first_deleted != -1:
                    return first_deleted
                return probe_index

            if self.keys[probe_index] is self.DELETED:
                if first_deleted == -1:
                    first_deleted = probe_index
                continue

            if self.keys[probe_index] == key:
                return probe_index

        return first_deleted if first_deleted != -1 else -1

    def put(self, key: Any, value: Any) -> bool:
        """키-값 삽입 - 평균 O(1), 최악 O(n)"""
        if self.count >= self.size * 0.7:  # 로드 팩터 70%
            self._resize()

        index = self._probe(key, for_insert=True)
        if index == -1:
            return False

        if self.keys[index] is None or self.keys[index] is self.DELETED:
            self.count += 1

        self.keys[index] = key
        self.values[index] = value
        return True

    def get(self, key: Any) -> Optional[Any]:
        """키로 값 조회 - 평균 O(1), 최악 O(n)"""
        index = self._probe(key)

        if index != -1 and self.keys[index] not in (None, self.DELETED):
            return self.values[index]
        return None

    def remove(self, key: Any) -> bool:
        """키-값 삭제 (삭제 마커 사용) - 평균 O(1), 최악 O(n)"""
        index = self._probe(key)

        if index != -1 and self.keys[index] not in (None, self.DELETED):
            self.keys[index] = self.DELETED
            self.values[index] = None
            self.count -= 1
            return True
        return False

    def _resize(self) -> None:
        """테이블 크기 2배 확장"""
        old_keys = self.keys
        old_values = self.values

        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)


# =============================================================================
# 4. 오픈 어드레싱 - 이중 해싱 (Double Hashing)
# =============================================================================

class HashTableDoubleHashing:
    """이중 해싱 방식 해시 테이블"""

    DELETED = object()

    def __init__(self, size: int = 11):  # 소수 권장
        self.size = size
        self.keys: List[Any] = [None] * size
        self.values: List[Any] = [None] * size
        self.count = 0

    def _hash1(self, key: Any) -> int:
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def _hash2(self, key: Any) -> int:
        """두 번째 해시 함수 (0이 아닌 값 반환)"""
        if isinstance(key, int):
            return 7 - (key % 7)  # 7은 size보다 작은 소수
        return 7 - (hash(key) % 7)

    def _probe(self, key: Any, for_insert: bool = False) -> int:
        """이중 해싱으로 슬롯 찾기"""
        h1 = self._hash1(key)
        h2 = self._hash2(key)
        first_deleted = -1

        for i in range(self.size):
            index = (h1 + i * h2) % self.size

            if self.keys[index] is None:
                if for_insert and first_deleted != -1:
                    return first_deleted
                return index

            if self.keys[index] is self.DELETED:
                if first_deleted == -1:
                    first_deleted = index
                continue

            if self.keys[index] == key:
                return index

        return first_deleted if first_deleted != -1 else -1

    def put(self, key: Any, value: Any) -> bool:
        index = self._probe(key, for_insert=True)
        if index == -1:
            return False

        if self.keys[index] is None or self.keys[index] is self.DELETED:
            self.count += 1

        self.keys[index] = key
        self.values[index] = value
        return True

    def get(self, key: Any) -> Optional[Any]:
        index = self._probe(key)
        if index != -1 and self.keys[index] not in (None, self.DELETED):
            return self.values[index]
        return None

    def remove(self, key: Any) -> bool:
        index = self._probe(key)
        if index != -1 and self.keys[index] not in (None, self.DELETED):
            self.keys[index] = self.DELETED
            self.values[index] = None
            self.count -= 1
            return True
        return False


# =============================================================================
# 5. LRU 캐시 (Least Recently Used Cache)
# =============================================================================

class LRUCache:
    """
    LRU 캐시 구현
    OrderedDict를 사용한 O(1) get/put
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        """키 조회 및 최근 사용으로 이동 - O(1)"""
        if key not in self.cache:
            return None

        # 최근 사용으로 이동
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """키-값 삽입 - O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # 가장 오래된 항목 제거
                self.cache.popitem(last=False)

        self.cache[key] = value

    def __str__(self) -> str:
        return str(list(self.cache.items()))


# =============================================================================
# 6. 실전 문제: Two Sum
# =============================================================================

def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    합이 target인 두 수의 인덱스 찾기
    시간복잡도: O(n), 공간복잡도: O(n)
    """
    seen = {}  # 값 -> 인덱스

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i

    return None


# =============================================================================
# 7. 실전 문제: 빈도 세기
# =============================================================================

def frequency_count(arr: List[Any]) -> dict:
    """
    배열 요소의 빈도 계산
    시간복잡도: O(n), 공간복잡도: O(k) (k = 고유 요소 수)
    """
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq


def most_frequent(arr: List[Any]) -> Optional[Any]:
    """가장 빈번한 요소 찾기"""
    freq = frequency_count(arr)
    if not freq:
        return None
    return max(freq, key=freq.get)


# =============================================================================
# 8. 실전 문제: 부분 배열 합이 K인 개수
# =============================================================================

def subarray_sum_count(nums: List[int], k: int) -> int:
    """
    합이 k인 연속 부분 배열 개수
    프리픽스 합 + 해시맵 활용
    시간복잡도: O(n), 공간복잡도: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # 프리픽스 합 -> 등장 횟수

    for num in nums:
        prefix_sum += num

        # prefix_sum - k가 이전에 있었다면, 그 사이가 합 k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count


# =============================================================================
# 9. 실전 문제: 아나그램 그룹
# =============================================================================

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    아나그램끼리 그룹화
    시간복잡도: O(n * k log k), n=문자열 수, k=최대 문자열 길이
    """
    groups = {}

    for s in strs:
        # 정렬된 문자열을 키로 사용
        key = ''.join(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    return list(groups.values())


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("해시 테이블 (Hash Table) 예제")
    print("=" * 60)

    # 1. 해시 함수 테스트
    print("\n[1] 해시 함수")
    print(f"    hash_division(42, 10) = {hash_division(42, 10)}")
    print(f"    hash_multiplication(42, 10) = {hash_multiplication(42, 10)}")
    print(f"    hash_string('hello', 10) = {hash_string('hello', 10)}")

    # 2. 체이닝 테스트
    print("\n[2] 체이닝 방식")
    ht_chain = HashTableChaining(5)
    for i, name in enumerate(['Alice', 'Bob', 'Charlie', 'David', 'Eve']):
        ht_chain.put(name, i * 10)
    print(f"    get('Charlie') = {ht_chain.get('Charlie')}")
    print(f"    'Bob' in table = {'Bob' in ht_chain}")
    print(f"    load_factor = {ht_chain.load_factor():.2f}")

    # 3. 선형 탐사 테스트
    print("\n[3] 선형 탐사 방식")
    ht_linear = HashTableLinearProbing(10)
    for i in range(7):
        ht_linear.put(i * 5, f"value_{i}")
    print(f"    get(10) = {ht_linear.get(10)}")
    ht_linear.remove(10)
    print(f"    get(10) after remove = {ht_linear.get(10)}")

    # 4. LRU 캐시 테스트
    print("\n[4] LRU 캐시")
    cache = LRUCache(3)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    print(f"    캐시: {cache}")
    cache.get('a')  # 'a'를 최근으로 이동
    cache.put('d', 4)  # 'b' 제거됨
    print(f"    get('a'), put('d') 후: {cache}")

    # 5. Two Sum
    print("\n[5] Two Sum")
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"    nums = {nums}, target = {target}")
    print(f"    결과: 인덱스 {result}")

    # 6. 빈도 세기
    print("\n[6] 빈도 세기")
    arr = ['a', 'b', 'a', 'c', 'a', 'b']
    freq = frequency_count(arr)
    print(f"    배열: {arr}")
    print(f"    빈도: {freq}")
    print(f"    최빈값: {most_frequent(arr)}")

    # 7. 부분 배열 합
    print("\n[7] 부분 배열 합이 K인 개수")
    nums = [1, 1, 1]
    k = 2
    count = subarray_sum_count(nums, k)
    print(f"    nums = {nums}, k = {k}")
    print(f"    결과: {count}개")

    # 8. 아나그램 그룹
    print("\n[8] 아나그램 그룹")
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = group_anagrams(strs)
    print(f"    입력: {strs}")
    print(f"    그룹: {groups}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
