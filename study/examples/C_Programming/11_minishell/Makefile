# Makefile for Mini Shell Examples

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
TARGET_DIR = .

# 실행 파일
TARGETS = minishell_v1 builtins redirect pipe minishell

# 기본 타겟
all: $(TARGETS)

# 1. 기본 쉘
minishell_v1: minishell_v1.c
	$(CC) $(CFLAGS) -o $(TARGET_DIR)/minishell_v1 minishell_v1.c

# 2. 내장 명령어 테스트
builtins: builtins.c
	$(CC) $(CFLAGS) -DTEST_BUILTINS -o $(TARGET_DIR)/builtins builtins.c

# 3. 리다이렉션 테스트
redirect: redirect.c
	$(CC) $(CFLAGS) -DTEST_REDIRECT -o $(TARGET_DIR)/redirect redirect.c

# 4. 파이프 테스트
pipe: pipe.c
	$(CC) $(CFLAGS) -DTEST_PIPE -o $(TARGET_DIR)/pipe pipe.c

# 5. 완성된 미니 쉘
minishell: minishell.c
	$(CC) $(CFLAGS) -o $(TARGET_DIR)/minishell minishell.c

# 테스트 (간단한 실행 확인)
test: all
	@echo "=== 컴파일 완료 ==="
	@echo "실행 파일:"
	@ls -lh minishell_v1 builtins redirect pipe minishell 2>/dev/null || true

# 정리
clean:
	rm -f $(TARGETS)
	rm -f *.o
	@echo "정리 완료"

# 도움말
help:
	@echo "Mini Shell Makefile"
	@echo ""
	@echo "사용법:"
	@echo "  make all          - 모든 실행 파일 컴파일"
	@echo "  make minishell_v1 - 기본 쉘만 컴파일"
	@echo "  make builtins     - 내장 명령어 테스트만 컴파일"
	@echo "  make redirect     - 리다이렉션 테스트만 컴파일"
	@echo "  make pipe         - 파이프 테스트만 컴파일"
	@echo "  make minishell    - 완성된 미니 쉘만 컴파일"
	@echo "  make test         - 컴파일 후 파일 확인"
	@echo "  make clean        - 실행 파일 제거"
	@echo "  make help         - 이 도움말 출력"

.PHONY: all test clean help
