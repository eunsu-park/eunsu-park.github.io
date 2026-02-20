"""
유니온 파인드 (Union-Find / Disjoint Set Union)
Union-Find / Disjoint Set Union Data Structure

서로소 집합을 관리하는 자료구조로, 그래프의 연결성 문제에 주로 사용됩니다.
"""

from typing import List, Tuple


# =============================================================================
# 1. 기본 유니온 파인드
# =============================================================================
class UnionFind:
    """
    기본 유니온 파인드 구현
    - 경로 압축 (Path Compression)
    - 랭크 기반 합치기 (Union by Rank)
    """

    def __init__(self, n: int):
        """
        n개의 요소로 초기화 (0 ~ n-1)
        """
        self.parent = list(range(n))  # 자기 자신을 부모로
        self.rank = [0] * n           # 트리의 높이 (근사값)
        self.count = n                # 집합의 개수

    def find(self, x: int) -> int:
        """
        x가 속한 집합의 대표(루트) 찾기
        경로 압축으로 거의 O(1)
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 경로 압축
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        x와 y가 속한 집합을 합치기
        이미 같은 집합이면 False 반환
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # 이미 같은 집합

        # 랭크 기반 합치기 (작은 트리를 큰 트리에 붙임)
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """x와 y가 같은 집합에 있는지 확인"""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """현재 집합의 개수"""
        return self.count


# =============================================================================
# 2. 크기 기반 유니온 파인드
# =============================================================================
class UnionFindWithSize:
    """
    집합의 크기를 추적하는 유니온 파인드
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n  # 각 집합의 크기

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # 크기 기반 합치기 (작은 집합을 큰 집합에)
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]

        return True

    def get_size(self, x: int) -> int:
        """x가 속한 집합의 크기"""
        return self.size[self.find(x)]


# =============================================================================
# 3. 연결 요소 개수
# =============================================================================
def count_components(n: int, edges: List[List[int]]) -> int:
    """
    n개의 노드와 간선 목록이 주어질 때 연결 요소 개수
    """
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.get_count()


# =============================================================================
# 4. 그래프에서 사이클 검출
# =============================================================================
def has_cycle(n: int, edges: List[List[int]]) -> bool:
    """
    무방향 그래프에서 사이클 존재 여부
    간선을 추가할 때 이미 같은 집합이면 사이클
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True  # 이미 연결됨 = 사이클
    return False


# =============================================================================
# 5. 크루스칼 MST (최소 신장 트리)
# =============================================================================
def kruskal_mst(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    크루스칼 알고리즘으로 최소 신장 트리 구하기
    edges: [(u, v, weight), ...]
    반환: (총 가중치, MST 간선 리스트)
    """
    # 가중치 기준 정렬
    edges = sorted(edges, key=lambda x: x[2])

    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))

            # n-1개의 간선을 선택하면 완료
            if len(mst_edges) == n - 1:
                break

    return mst_weight, mst_edges


# =============================================================================
# 6. 친구 관계 (계정 병합)
# =============================================================================
def merge_accounts(accounts: List[List[str]]) -> List[List[str]]:
    """
    같은 이메일을 가진 계정 병합
    accounts[i] = [이름, 이메일1, 이메일2, ...]
    """
    from collections import defaultdict

    # 이메일 -> 계정 인덱스 매핑
    email_to_id = {}
    email_to_name = {}

    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            if email in email_to_id:
                pass  # 나중에 union
            email_to_id[email] = i
            email_to_name[email] = name

    # 유니온 파인드로 같은 사람의 계정 연결
    n = len(accounts)
    uf = UnionFind(n)

    email_first_account = {}
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_first_account:
                uf.union(i, email_first_account[email])
            else:
                email_first_account[email] = i

    # 결과 집계
    root_to_emails = defaultdict(set)
    for i, account in enumerate(accounts):
        root = uf.find(i)
        for email in account[1:]:
            root_to_emails[root].add(email)

    # 결과 포맷팅
    result = []
    for root, emails in root_to_emails.items():
        name = accounts[root][0]
        result.append([name] + sorted(emails))

    return result


# =============================================================================
# 7. 섬 연결하기 (2D 그리드)
# =============================================================================
def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    '1'은 땅, '0'은 물
    연결된 땅 덩어리(섬)의 개수
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])

    # 2D -> 1D 좌표 변환
    def get_index(r, c):
        return r * cols + c

    uf = UnionFind(rows * cols)
    land_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                land_count += 1
                # 오른쪽, 아래 방향만 확인 (중복 방지)
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        if uf.union(get_index(r, c), get_index(nr, nc)):
                            land_count -= 1

    return land_count


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("유니온 파인드 (Union-Find) 예제")
    print("=" * 60)

    # 1. 기본 사용
    print("\n[1] 기본 유니온 파인드")
    uf = UnionFind(10)
    operations = [(0, 1), (2, 3), (4, 5), (1, 2), (6, 7), (8, 9), (0, 9)]
    for u, v in operations:
        uf.union(u, v)
        print(f"    union({u}, {v}) -> 집합 수: {uf.get_count()}")

    print(f"\n    0과 9 연결됨? {uf.connected(0, 9)}")
    print(f"    0과 6 연결됨? {uf.connected(0, 6)}")

    # 2. 크기 추적
    print("\n[2] 크기 기반 유니온 파인드")
    uf_size = UnionFindWithSize(5)
    uf_size.union(0, 1)
    uf_size.union(2, 3)
    uf_size.union(0, 2)
    print(f"    0이 속한 집합 크기: {uf_size.get_size(0)}")
    print(f"    4가 속한 집합 크기: {uf_size.get_size(4)}")

    # 3. 연결 요소 개수
    print("\n[3] 연결 요소 개수")
    edges = [[0, 1], [1, 2], [3, 4]]
    count = count_components(5, edges)
    print(f"    노드 5개, 간선: {edges}")
    print(f"    연결 요소 개수: {count}")

    # 4. 사이클 검출
    print("\n[4] 사이클 검출")
    edges_no_cycle = [[0, 1], [1, 2], [2, 3]]
    edges_with_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    간선 {edges_no_cycle}: 사이클 = {has_cycle(4, edges_no_cycle)}")
    print(f"    간선 {edges_with_cycle}: 사이클 = {has_cycle(3, edges_with_cycle)}")

    # 5. 크루스칼 MST
    print("\n[5] 크루스칼 MST")
    #     1
    #   0---1
    #   |\  |
    # 4 | \ |2
    #   |  \|
    #   3---2
    #     3
    edges_mst = [
        (0, 1, 1), (0, 2, 4), (0, 3, 4),
        (1, 2, 2), (2, 3, 3)
    ]
    total_weight, mst_edges = kruskal_mst(4, edges_mst)
    print(f"    간선: {edges_mst}")
    print(f"    MST 총 가중치: {total_weight}")
    print(f"    MST 간선: {mst_edges}")

    # 6. 계정 병합
    print("\n[6] 계정 병합")
    accounts = [
        ["John", "john@mail.com", "john_work@mail.com"],
        ["John", "john@mail.com", "john2@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "john3@mail.com"]
    ]
    result = merge_accounts(accounts)
    print(f"    입력:")
    for acc in accounts:
        print(f"      {acc}")
    print(f"    병합 결과:")
    for acc in result:
        print(f"      {acc}")

    # 7. 섬 개수 (유니온 파인드)
    print("\n[7] 섬의 개수 (유니온 파인드)")
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    count = num_islands_union_find(grid)
    print(f"    격자:")
    for row in grid:
        print(f"    {row}")
    print(f"    섬의 개수: {count}")

    print("\n" + "=" * 60)
    print("유니온 파인드 시간 복잡도")
    print("=" * 60)
    print("""
    경로 압축 + 랭크/크기 기반 합치기 사용 시:
    - find(): 거의 O(1) (정확히는 O(α(n)), α는 아커만 역함수)
    - union(): 거의 O(1)
    - 공간 복잡도: O(n)

    주요 활용:
    - 연결 요소 관리
    - 사이클 검출
    - 최소 신장 트리 (크루스칼)
    - 동적 연결성 문제
    """)


if __name__ == "__main__":
    main()
