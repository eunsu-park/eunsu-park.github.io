# Makefile for Hash Table Project
# 해시 테이블 프로젝트 빌드 파일

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
DEBUG_FLAGS = -g -DDEBUG

# 타겟 실행 파일
TARGETS = hash_functions hash_chaining hash_linear_probing dictionary

# 기본 타겟: 모든 프로그램 컴파일
all: $(TARGETS)

# 개별 프로그램 컴파일
hash_functions: hash_functions.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_functions 컴파일 완료"

hash_chaining: hash_chaining.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_chaining 컴파일 완료"

hash_linear_probing: hash_linear_probing.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_linear_probing 컴파일 완료"

dictionary: dictionary.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ dictionary 컴파일 완료"

# 디버그 버전 컴파일
debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "✓ 디버그 버전으로 컴파일 완료"

# 실행 타겟
run_functions: hash_functions
	@echo "=== 해시 함수 비교 실행 ==="
	./hash_functions

run_chaining: hash_chaining
	@echo "=== 체이닝 해시 테이블 실행 ==="
	./hash_chaining

run_probing: hash_linear_probing
	@echo "=== 선형 탐사 해시 테이블 실행 ==="
	./hash_linear_probing

run_dict: dictionary
	@echo "=== 사전 프로그램 실행 ==="
	./dictionary

# 모든 프로그램 순차 실행 (테스트용)
test: all
	@echo "\n[1/4] 해시 함수 비교 테스트"
	@./hash_functions | head -50
	@echo "\n[2/4] 체이닝 테스트"
	@./hash_chaining | head -30
	@echo "\n[3/4] 선형 탐사 테스트"
	@./hash_linear_probing | head -30
	@echo "\n[4/4] 모든 테스트 완료"

# 정리
clean:
	rm -f $(TARGETS)
	rm -f *.o
	rm -f dictionary.txt
	@echo "✓ 생성된 파일 삭제 완료"

# 코드 품질 검사 (정적 분석)
check:
	@echo "=== 코드 품질 검사 ==="
	@for file in *.c; do \
		echo "Checking $$file..."; \
		$(CC) $(CFLAGS) -fsyntax-only $$file; \
	done
	@echo "✓ 모든 파일 문법 검사 완료"

# 메모리 누수 검사 (valgrind 필요)
memcheck: all
	@echo "=== 메모리 누수 검사 (valgrind) ==="
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --leak-check=full --show-leak-kinds=all ./hash_chaining; \
		valgrind --leak-check=full --show-leak-kinds=all ./hash_linear_probing; \
	else \
		echo "valgrind가 설치되지 않았습니다"; \
	fi

# 도움말
help:
	@echo "해시 테이블 프로젝트 Makefile"
	@echo ""
	@echo "사용 가능한 타겟:"
	@echo "  make              - 모든 프로그램 컴파일"
	@echo "  make debug        - 디버그 모드로 컴파일"
	@echo "  make clean        - 생성된 파일 삭제"
	@echo "  make test         - 모든 프로그램 테스트 실행"
	@echo "  make check        - 코드 품질 검사"
	@echo "  make memcheck     - 메모리 누수 검사 (valgrind)"
	@echo ""
	@echo "개별 실행:"
	@echo "  make run_functions  - 해시 함수 비교 실행"
	@echo "  make run_chaining   - 체이닝 해시 테이블 실행"
	@echo "  make run_probing    - 선형 탐사 해시 테이블 실행"
	@echo "  make run_dict       - 사전 프로그램 실행"

.PHONY: all debug clean test check memcheck help run_functions run_chaining run_probing run_dict
