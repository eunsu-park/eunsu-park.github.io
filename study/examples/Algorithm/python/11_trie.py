"""
트라이 (Trie / Prefix Tree)
Trie Data Structure

문자열 검색과 접두사 기반 연산을 위한 트라이를 구현합니다.
"""

from typing import List, Optional, Dict
from collections import defaultdict


# =============================================================================
# 1. 기본 트라이 (딕셔너리 기반)
# =============================================================================

class TrieNode:
    """트라이 노드"""

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.count: int = 0  # 이 접두사로 시작하는 단어 수


class Trie:
    """
    트라이 (접두사 트리)
    - 삽입: O(m), m = 단어 길이
    - 검색: O(m)
    - 접두사 검색: O(m)
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """단어 삽입 - O(m)"""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1

        node.is_end = True

    def search(self, word: str) -> bool:
        """단어 존재 여부 검색 - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """접두사로 시작하는 단어 존재 여부 - O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """접두사로 시작하는 단어 개수 - O(m)"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """접두사에 해당하는 노드 찾기"""
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node

    def delete(self, word: str) -> bool:
        """단어 삭제 - O(m)"""

        def _delete(node: TrieNode, word: str, depth: int) -> bool:
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            should_delete = _delete(node.children[char], word, depth + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end

            node.children[char].count -= 1
            return False

        return _delete(self.root, word, 0)


# =============================================================================
# 2. 자동완성 (Autocomplete)
# =============================================================================

class AutocompleteSystem:
    """자동완성 시스템"""

    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str, weight: int = 1) -> None:
        """단어 추가 (가중치 포함)"""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True
        node.count = weight  # 여기서 count는 빈도/가중치

    def autocomplete(self, prefix: str, limit: int = 5) -> List[str]:
        """접두사로 시작하는 단어 목록 - O(m + k), k = 결과 수"""
        node = self.root

        # 접두사 노드 찾기
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS로 모든 단어 수집
        results = []
        self._collect_words(node, prefix, results)

        # 빈도순 정렬 후 상위 limit개
        results.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in results[:limit]]

    def _collect_words(self, node: TrieNode, prefix: str, results: List) -> None:
        """노드에서 모든 완성 단어 수집"""
        if node.is_end:
            results.append((prefix, node.count))

        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)


# =============================================================================
# 3. 와일드카드 검색
# =============================================================================

class WildcardTrie:
    """와일드카드 '.'를 지원하는 트라이"""

    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """'.'는 임의의 한 문자와 매칭"""
        return self._search(self.root, word, 0)

    def _search(self, node: TrieNode, word: str, index: int) -> bool:
        if index == len(word):
            return node.is_end

        char = word[index]

        if char == '.':
            # 모든 자식 노드 시도
            for child in node.children.values():
                if self._search(child, word, index + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return self._search(node.children[char], word, index + 1)


# =============================================================================
# 4. 최장 공통 접두사 (LCP)
# =============================================================================

def longest_common_prefix(words: List[str]) -> str:
    """
    단어 배열의 최장 공통 접두사
    트라이 활용
    """
    if not words:
        return ""

    # 트라이 구성
    trie = Trie()
    for word in words:
        trie.insert(word)

    # 루트에서 분기점까지 이동
    prefix = []
    node = trie.root

    while node:
        # 자식이 하나이고 종료 노드가 아닐 때만 계속
        if len(node.children) == 1 and not node.is_end:
            char = list(node.children.keys())[0]
            prefix.append(char)
            node = node.children[char]
        else:
            break

    return ''.join(prefix)


# =============================================================================
# 5. 단어 사전 (Word Dictionary)
# =============================================================================

class WordDictionary:
    """
    단어 추가/검색 사전
    - 정확한 검색
    - 접두사 검색
    - 와일드카드 검색
    """

    def __init__(self):
        self.trie = Trie()
        self.wildcard_trie = WildcardTrie()

    def add_word(self, word: str) -> None:
        self.trie.insert(word)
        self.wildcard_trie.add_word(word)

    def search_exact(self, word: str) -> bool:
        """정확한 단어 검색"""
        return self.trie.search(word)

    def search_prefix(self, prefix: str) -> bool:
        """접두사 검색"""
        return self.trie.starts_with(prefix)

    def search_pattern(self, pattern: str) -> bool:
        """패턴 검색 (. = 임의 문자)"""
        return self.wildcard_trie.search(pattern)


# =============================================================================
# 6. XOR 트라이 (최대 XOR 쌍 찾기)
# =============================================================================

class XORTrie:
    """
    비트 트라이로 최대 XOR 쌍 찾기
    각 숫자를 이진수로 저장
    """

    def __init__(self, max_bits: int = 31):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num: int) -> None:
        """숫자 삽입 - O(max_bits)"""
        node = self.root

        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num: int) -> int:
        """num과 XOR했을 때 최대가 되는 값 반환 - O(max_bits)"""
        node = self.root
        result = 0

        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            # XOR 최대화를 위해 반대 비트 선택
            toggle_bit = 1 - bit

            if toggle_bit in node:
                result |= (1 << i)
                node = node[toggle_bit]
            elif bit in node:
                node = node[bit]
            else:
                break

        return result


def find_maximum_xor(nums: List[int]) -> int:
    """
    배열에서 두 수의 XOR 최댓값
    시간복잡도: O(n * max_bits)
    """
    if len(nums) < 2:
        return 0

    xor_trie = XORTrie()
    max_xor = 0

    for num in nums:
        xor_trie.insert(num)
        max_xor = max(max_xor, xor_trie.find_max_xor(num))

    return max_xor


# =============================================================================
# 7. 실전 문제: 단어 검색 II (Word Search II)
# =============================================================================

def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    2D 보드에서 단어 목록 찾기
    트라이 + DFS
    """
    # 트라이 구성
    trie = Trie()
    for word in words:
        trie.insert(word)

    rows, cols = len(board), len(board[0])
    result = set()

    def dfs(r: int, c: int, node: TrieNode, path: str) -> None:
        if node.is_end:
            result.add(path)

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        char = board[r][c]
        if char == '#' or char not in node.children:
            return

        board[r][c] = '#'  # 방문 표시
        next_node = node.children[char]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(r + dr, c + dc, next_node, path + char)

        board[r][c] = char  # 복원

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, "")

    return list(result)


# =============================================================================
# 8. 접두사/접미사 동시 검색
# =============================================================================

class PrefixSuffixTrie:
    """접두사와 접미사를 동시에 검색하는 트라이"""

    def __init__(self, words: List[str]):
        self.trie = {}

        for idx, word in enumerate(words):
            # 모든 접미사#단어 형태로 저장
            key = word + '#' + word
            for i in range(len(word) + 1):
                node = self.trie
                for char in key[i:]:
                    if char not in node:
                        node[char] = {'idx': -1}
                    node = node[char]
                    node['idx'] = idx  # 가장 큰 인덱스 저장

    def search(self, prefix: str, suffix: str) -> int:
        """접두사와 접미사 모두 만족하는 단어의 인덱스"""
        key = suffix + '#' + prefix
        node = self.trie

        for char in key:
            if char not in node:
                return -1
            node = node[char]

        return node.get('idx', -1)


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("트라이 (Trie / Prefix Tree) 예제")
    print("=" * 60)

    # 1. 기본 트라이
    print("\n[1] 기본 트라이")
    trie = Trie()
    words = ["apple", "app", "application", "banana", "band"]
    for word in words:
        trie.insert(word)
    print(f"    삽입: {words}")
    print(f"    search('app'): {trie.search('app')}")
    print(f"    search('application'): {trie.search('application')}")
    print(f"    search('apply'): {trie.search('apply')}")
    print(f"    starts_with('app'): {trie.starts_with('app')}")
    print(f"    count_prefix('app'): {trie.count_prefix('app')}")

    # 2. 자동완성
    print("\n[2] 자동완성")
    auto = AutocompleteSystem()
    search_words = [("hello", 5), ("help", 3), ("helicopter", 2), ("hero", 4), ("world", 1)]
    for word, weight in search_words:
        auto.add_word(word, weight)
    print(f"    단어/빈도: {search_words}")
    print(f"    'hel' 자동완성: {auto.autocomplete('hel')}")
    print(f"    'he' 자동완성: {auto.autocomplete('he')}")

    # 3. 와일드카드 검색
    print("\n[3] 와일드카드 검색")
    wild = WildcardTrie()
    for word in ["bad", "dad", "mad", "pad"]:
        wild.add_word(word)
    print(f"    단어: ['bad', 'dad', 'mad', 'pad']")
    print(f"    search('.ad'): {wild.search('.ad')}")
    print(f"    search('b..'): {wild.search('b..')}")
    print(f"    search('..d'): {wild.search('..d')}")
    print(f"    search('b.d'): {wild.search('b.d')}")

    # 4. 최장 공통 접두사
    print("\n[4] 최장 공통 접두사")
    words_lcp = ["flower", "flow", "flight"]
    lcp = longest_common_prefix(words_lcp)
    print(f"    단어: {words_lcp}")
    print(f"    LCP: '{lcp}'")

    # 5. XOR 트라이
    print("\n[5] 최대 XOR (XOR 트라이)")
    nums = [3, 10, 5, 25, 2, 8]
    max_xor = find_maximum_xor(nums)
    print(f"    배열: {nums}")
    print(f"    최대 XOR: {max_xor}")
    print(f"    (5 XOR 25 = {5 ^ 25})")

    # 6. 단어 사전
    print("\n[6] 단어 사전")
    dictionary = WordDictionary()
    for word in ["hello", "help", "world"]:
        dictionary.add_word(word)
    print(f"    단어: ['hello', 'help', 'world']")
    print(f"    exact 'help': {dictionary.search_exact('help')}")
    print(f"    prefix 'hel': {dictionary.search_prefix('hel')}")
    print(f"    pattern 'h.l.o': {dictionary.search_pattern('h.l.o')}")

    # 7. 단어 삭제
    print("\n[7] 단어 삭제")
    trie2 = Trie()
    for word in ["apple", "app"]:
        trie2.insert(word)
    print(f"    삽입: ['apple', 'app']")
    print(f"    search('app'): {trie2.search('app')}")
    trie2.delete("app")
    print(f"    delete('app') 후")
    print(f"    search('app'): {trie2.search('app')}")
    print(f"    search('apple'): {trie2.search('apple')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
