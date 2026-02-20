# 전체 예제 빌드 Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
THREAD_FLAGS = -pthread

# C 프로그램 목록
C_PROGRAMS = \
	02_calculator/calculator \
	03_number_guess/number_guess \
	05_dynamic_array/dynamic_array \
	06_linked_list/linked_list \
	14_bit_operations/bit_manipulation

# 멀티스레드 프로그램
THREAD_PROGRAMS = \
	12_multithread/thread_basic

.PHONY: all clean c-programs thread-programs

all: c-programs thread-programs

c-programs:
	@echo "=== C 프로그램 컴파일 ==="
	@for prog in $(C_PROGRAMS); do \
		echo "Compiling $$prog..."; \
		$(CC) $(CFLAGS) $${prog}.c -o $$prog || exit 1; \
	done
	@echo "완료!"

thread-programs:
	@echo ""
	@echo "=== 멀티스레드 프로그램 컴파일 ==="
	@for prog in $(THREAD_PROGRAMS); do \
		echo "Compiling $$prog..."; \
		$(CC) $(CFLAGS) $(THREAD_FLAGS) $${prog}.c -o $$prog || exit 1; \
	done
	@echo "완료!"

clean:
	@echo "=== 정리 중 ==="
	@for prog in $(C_PROGRAMS) $(THREAD_PROGRAMS); do \
		rm -f $$prog; \
	done
	@echo "완료!"

# 개별 프로그램 실행
run-calculator:
	@echo "=== 계산기 실행 ==="
	@./02_calculator/calculator

run-guess:
	@echo "=== 숫자 맞추기 실행 ==="
	@./03_number_guess/number_guess

run-array:
	@echo "=== 동적 배열 실행 ==="
	@./05_dynamic_array/dynamic_array

run-list:
	@echo "=== 연결 리스트 실행 ==="
	@./06_linked_list/linked_list

run-bit:
	@echo "=== 비트 연산 실행 ==="
	@./14_bit_operations/bit_manipulation

run-thread:
	@echo "=== 스레드 실행 ==="
	@./12_multithread/thread_basic

# 도움말
help:
	@echo "사용 가능한 명령:"
	@echo "  make              - 모든 프로그램 컴파일"
	@echo "  make c-programs   - C 프로그램만 컴파일"
	@echo "  make thread-programs - 멀티스레드 프로그램 컴파일"
	@echo "  make clean        - 컴파일된 파일 삭제"
	@echo ""
	@echo "개별 실행:"
	@echo "  make run-calculator  - 계산기 실행"
	@echo "  make run-guess       - 숫자 맞추기 실행"
	@echo "  make run-array       - 동적 배열 실행"
	@echo "  make run-list        - 연결 리스트 실행"
	@echo "  make run-bit         - 비트 연산 실행"
	@echo "  make run-thread      - 스레드 실행"
