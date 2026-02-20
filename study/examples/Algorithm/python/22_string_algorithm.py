"""
문자열 알고리즘 (String Algorithms)
String Matching and Processing

문자열 검색 및 처리 알고리즘을 구현합니다.
"""

from typing import List, Tuple


# =============================================================================
# 1. KMP 알고리즘 (Knuth-Morris-Pratt)
# =============================================================================

def kmp_failure(pattern: str) -> List[int]:
    """
    KMP 실패 함수 (부분 일치 테이블) 계산
    시간복잡도: O(m), m = 패턴 길이
    """
    m = len(pattern)
    failure = [0] * m
    j = 0  # 이전 최대 접두사 길이

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]

        if pattern[i] == pattern[j]:
            j += 1
            failure[i] = j

    return failure


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    KMP 문자열 검색
    시간복잡도: O(n + m)
    반환: 패턴이 발견된 시작 인덱스 리스트
    """
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    failure = kmp_failure(pattern)
    matches = []
    j = 0  # 패턴 인덱스

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]

        if text[i] == pattern[j]:
            if j == m - 1:
                matches.append(i - m + 1)
                j = failure[j]
            else:
                j += 1

    return matches


# =============================================================================
# 2. Rabin-Karp 알고리즘
# =============================================================================

def rabin_karp_search(text: str, pattern: str, mod: int = 10**9 + 7) -> List[int]:
    """
    Rabin-Karp 문자열 검색 (롤링 해시)
    시간복잡도: 평균 O(n + m), 최악 O(nm)
    """
    if not pattern or len(pattern) > len(text):
        return []

    n, m = len(text), len(pattern)
    base = 256
    matches = []

    # 패턴 해시 계산
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # 해시 충돌 확인
            if text[i:i + m] == pattern:
                matches.append(i)

        # 다음 윈도우 해시 계산
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * h) * base + ord(text[i + m])) % mod

    return matches


# =============================================================================
# 3. Z 알고리즘
# =============================================================================

def z_function(s: str) -> List[int]:
    """
    Z 함수 계산
    z[i] = s와 s[i:]의 최장 공통 접두사 길이
    시간복잡도: O(n)
    """
    n = len(s)
    z = [0] * n
    z[0] = n

    l, r = 0, 0  # Z-box [l, r)

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z


def z_search(text: str, pattern: str) -> List[int]:
    """Z 알고리즘을 이용한 문자열 검색"""
    if not pattern:
        return []

    combined = pattern + "$" + text
    z = z_function(combined)
    m = len(pattern)

    return [i - m - 1 for i in range(m + 1, len(combined)) if z[i] == m]


# =============================================================================
# 4. Manacher 알고리즘 (최장 회문 부분문자열)
# =============================================================================

def manacher(s: str) -> Tuple[int, int]:
    """
    최장 회문 부분문자열 찾기
    시간복잡도: O(n)
    반환: (시작 인덱스, 길이)
    """
    if not s:
        return 0, 0

    # 전처리: 문자 사이에 # 삽입
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = i 중심 회문 반지름

    c, r = 0, 0  # 현재 회문 중심, 오른쪽 경계

    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])

        # 확장 시도
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        # 경계 업데이트
        if i + p[i] > r:
            c, r = i, i + p[i]

    # 최장 회문 찾기
    max_len = max(p)
    center = p.index(max_len)

    # 원본 문자열에서의 위치
    start = (center - max_len) // 2
    length = max_len

    return start, length


def longest_palindrome(s: str) -> str:
    """최장 회문 부분문자열 반환"""
    start, length = manacher(s)
    return s[start:start + length]


# =============================================================================
# 5. 접미사 배열 (Suffix Array) - 간단 구현
# =============================================================================

def suffix_array(s: str) -> List[int]:
    """
    접미사 배열 생성 (간단한 O(n log² n) 구현)
    반환: 사전순 정렬된 접미사의 시작 인덱스
    """
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        # (rank[i], rank[i+k]) 기준 정렬
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)

        sa.sort(key=key)

        # 새 rank 계산
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if key(sa[i]) != key(sa[i - 1]):
                tmp[sa[i]] += 1

        rank = tmp[:]
        k *= 2

    return sa


def lcp_array(s: str, sa: List[int]) -> List[int]:
    """
    LCP 배열 (Longest Common Prefix)
    lcp[i] = s[sa[i]:]와 s[sa[i+1]:]의 최장 공통 접두사 길이
    시간복잡도: O(n)
    """
    n = len(s)
    rank = [0] * n
    for i, idx in enumerate(sa):
        rank[idx] = i

    lcp = [0] * (n - 1)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1

    return lcp


# =============================================================================
# 6. 트라이 기반 문자열 검색
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.output = []  # Aho-Corasick용


class AhoCorasick:
    """
    Aho-Corasick 알고리즘 (다중 패턴 검색)
    전처리: O(Σ|patterns|)
    검색: O(n + m), m = 매칭 수
    """

    def __init__(self, patterns: List[str]):
        self.root = TrieNode()
        self.patterns = patterns
        self._build_trie()
        self._build_failure()

    def _build_trie(self):
        for idx, pattern in enumerate(self.patterns):
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.output.append(idx)

    def _build_failure(self):
        from collections import deque

        queue = deque()
        self.root.fail = self.root

        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            node = queue.popleft()

            for char, child in node.children.items():
                fail = node.fail
                while fail != self.root and char not in fail.children:
                    fail = fail.fail

                child.fail = fail.children.get(char, self.root)
                if child.fail == child:
                    child.fail = self.root

                child.output += child.fail.output
                queue.append(child)

    def search(self, text: str) -> List[Tuple[int, int]]:
        """
        텍스트에서 모든 패턴 검색
        반환: [(위치, 패턴 인덱스), ...]
        """
        results = []
        node = self.root

        for i, char in enumerate(text):
            while node != self.root and char not in node.children:
                node = node.fail

            node = node.children.get(char, self.root)

            for pattern_idx in node.output:
                pattern = self.patterns[pattern_idx]
                results.append((i - len(pattern) + 1, pattern_idx))

        return results


# =============================================================================
# 7. 편집 거리 (Edit Distance)
# =============================================================================

def edit_distance(s1: str, s2: str) -> int:
    """
    레벤슈타인 거리 (편집 거리)
    시간복잡도: O(mn)
    공간복잡도: O(min(m, n)) 최적화 가능
    """
    m, n = len(s1), len(s2)

    # 공간 최적화: 두 행만 사용
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i

        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])

        prev, curr = curr, prev

    return prev[n]


# =============================================================================
# 8. 문자열 해싱
# =============================================================================

class StringHash:
    """
    다항식 롤링 해시
    충돌을 줄이기 위해 두 개의 해시 사용 (double hashing)
    """

    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.MOD1 = 10**9 + 7
        self.MOD2 = 10**9 + 9
        self.BASE1 = 31
        self.BASE2 = 37

        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash1[i + 1] = (self.hash1[i] * self.BASE1 + ord(s[i])) % self.MOD1
            self.hash2[i + 1] = (self.hash2[i] * self.BASE2 + ord(s[i])) % self.MOD2
            self.pow1[i + 1] = self.pow1[i] * self.BASE1 % self.MOD1
            self.pow2[i + 1] = self.pow2[i] * self.BASE2 % self.MOD2

    def get_hash(self, l: int, r: int) -> Tuple[int, int]:
        """s[l:r]의 해시 값 (0-indexed, 반열린 구간)"""
        h1 = (self.hash1[r] - self.hash1[l] * self.pow1[r - l]) % self.MOD1
        h2 = (self.hash2[r] - self.hash2[l] * self.pow2[r - l]) % self.MOD2
        return (h1, h2)

    def is_equal(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """두 부분문자열이 같은지 확인"""
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("문자열 알고리즘 (String Algorithms) 예제")
    print("=" * 60)

    # 1. KMP
    print("\n[1] KMP 알고리즘")
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    failure = kmp_failure(pattern)
    matches = kmp_search(text, pattern)
    print(f"    텍스트: {text}")
    print(f"    패턴: {pattern}")
    print(f"    실패 함수: {failure}")
    print(f"    매칭 위치: {matches}")

    # 2. Rabin-Karp
    print("\n[2] Rabin-Karp 알고리즘")
    matches = rabin_karp_search(text, pattern)
    print(f"    매칭 위치: {matches}")

    # 3. Z 알고리즘
    print("\n[3] Z 알고리즘")
    s = "aabxaab"
    z = z_function(s)
    print(f"    문자열: {s}")
    print(f"    Z 배열: {z}")
    matches = z_search(text, pattern)
    print(f"    검색 결과: {matches}")

    # 4. Manacher
    print("\n[4] Manacher 알고리즘 (최장 회문)")
    s = "babad"
    palindrome = longest_palindrome(s)
    print(f"    문자열: {s}")
    print(f"    최장 회문: {palindrome}")

    s2 = "abacdfgdcaba"
    palindrome2 = longest_palindrome(s2)
    print(f"    문자열: {s2}")
    print(f"    최장 회문: {palindrome2}")

    # 5. 접미사 배열
    print("\n[5] 접미사 배열")
    s = "banana"
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    print(f"    문자열: {s}")
    print(f"    접미사 배열: {sa}")
    print("    접미사들:")
    for i in sa:
        print(f"      {i}: {s[i:]}")
    print(f"    LCP 배열: {lcp}")

    # 6. Aho-Corasick
    print("\n[6] Aho-Corasick (다중 패턴)")
    patterns = ["he", "she", "his", "hers"]
    text = "ahishers"
    ac = AhoCorasick(patterns)
    results = ac.search(text)
    print(f"    패턴: {patterns}")
    print(f"    텍스트: {text}")
    print("    매칭:")
    for pos, idx in results:
        print(f"      위치 {pos}: '{patterns[idx]}'")

    # 7. 편집 거리
    print("\n[7] 편집 거리")
    s1, s2 = "kitten", "sitting"
    dist = edit_distance(s1, s2)
    print(f"    '{s1}' → '{s2}'")
    print(f"    편집 거리: {dist}")

    # 8. 문자열 해싱
    print("\n[8] 문자열 해싱")
    s = "abcabc"
    sh = StringHash(s)
    print(f"    문자열: {s}")
    print(f"    hash(0:3) = {sh.get_hash(0, 3)}")
    print(f"    hash(3:6) = {sh.get_hash(3, 6)}")
    print(f"    s[0:3] == s[3:6]: {sh.is_equal(0, 3, 3, 6)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
