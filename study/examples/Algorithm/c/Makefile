# Algorithm Examples - C Makefile
# 알고리즘 예제 C 버전 빌드용

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lm

# 모든 소스 파일
SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)

# 기본 타겟
.PHONY: all clean help test run-all

all: $(TARGETS)

# 개별 빌드 규칙
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# 정리
clean:
	rm -f $(TARGETS)
	rm -f *.o

# 도움말
help:
	@echo "Algorithm Examples - C Makefile"
	@echo ""
	@echo "사용법:"
	@echo "  make          - 모든 예제 컴파일"
	@echo "  make clean    - 실행 파일 삭제"
	@echo "  make test     - 모든 예제 실행 및 테스트"
	@echo "  make run-XX   - 특정 예제 실행 (예: make run-01)"
	@echo ""
	@echo "예제 목록:"
	@echo "  01_complexity      - 시간 복잡도"
	@echo "  02_array_string    - 배열과 문자열"
	@echo "  03_stack_queue     - 스택과 큐"
	@echo "  04_hash_table      - 해시 테이블"
	@echo "  05_sorting         - 정렬 알고리즘"
	@echo "  06_searching       - 탐색 알고리즘"
	@echo "  07_divide_conquer  - 분할 정복"
	@echo "  08_backtracking    - 백트래킹"
	@echo "  09_tree_bst        - 트리와 BST"
	@echo "  10_heap            - 힙"
	@echo "  11_trie            - 트라이"
	@echo "  12_graph_basic     - 그래프 기초"
	@echo "  13_topological_sort- 위상 정렬"
	@echo "  14_shortest_path   - 최단 경로"
	@echo "  15_mst             - 최소 신장 트리"
	@echo "  16_lca             - LCA"
	@echo "  17_scc             - SCC"
	@echo "  18_dp              - 동적 프로그래밍"
	@echo "  19_greedy          - 탐욕 알고리즘"
	@echo "  20_bitmask_dp      - 비트마스크 DP"
	@echo "  21_number_theory   - 정수론"
	@echo "  22_string_algorithm- 문자열 알고리즘"
	@echo "  23_segment_tree    - 세그먼트 트리"
	@echo "  24_fenwick_tree    - 펜윅 트리"
	@echo "  25_network_flow    - 네트워크 플로우"
	@echo "  26_geometry        - 기하 알고리즘"
	@echo "  27_game_theory     - 게임 이론"
	@echo "  28_advanced_dp     - 고급 DP 최적화"
	@echo "  29_practice        - 실전 문제 풀이"

# 테스트 (모든 예제 실행)
test: all
	@echo "=========================================="
	@echo "모든 예제 테스트 실행"
	@echo "=========================================="
	@for target in $(TARGETS); do \
		echo ""; \
		echo "▶ Running $$target..."; \
		./$$target || echo "✗ $$target 실패"; \
		echo ""; \
	done
	@echo "=========================================="
	@echo "테스트 완료"
	@echo "=========================================="

# 전체 실행
run-all: all
	@for target in $(TARGETS); do \
		./$$target; \
	done

# 개별 실행 규칙
run-01: 01_complexity
	./01_complexity

run-02: 02_array_string
	./02_array_string

run-03: 03_stack_queue
	./03_stack_queue

run-04: 04_hash_table
	./04_hash_table

run-05: 05_sorting
	./05_sorting

run-06: 06_searching
	./06_searching

run-07: 07_divide_conquer
	./07_divide_conquer

run-08: 08_backtracking
	./08_backtracking

run-09: 09_tree_bst
	./09_tree_bst

run-10: 10_heap
	./10_heap

run-11: 11_trie
	./11_trie

run-12: 12_graph_basic
	./12_graph_basic

run-13: 13_topological_sort
	./13_topological_sort

run-14: 14_shortest_path
	./14_shortest_path

run-15: 15_mst
	./15_mst

run-16: 16_lca
	./16_lca

run-17: 17_scc
	./17_scc

run-18: 18_dp
	./18_dp

run-19: 19_greedy
	./19_greedy

run-20: 20_bitmask_dp
	./20_bitmask_dp

run-21: 21_number_theory
	./21_number_theory

run-22: 22_string_algorithm
	./22_string_algorithm

run-23: 23_segment_tree
	./23_segment_tree

run-24: 24_fenwick_tree
	./24_fenwick_tree

run-25: 25_network_flow
	./25_network_flow

run-26: 26_geometry
	./26_geometry

run-27: 27_game_theory
	./27_game_theory

run-28: 28_advanced_dp
	./28_advanced_dp

run-29: 29_practice
	./29_practice
