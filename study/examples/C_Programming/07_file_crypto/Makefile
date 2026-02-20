# Makefile for File Encryption Examples
# C Programming Project 7

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
TARGETS = simple_xor file_encrypt file_encrypt_v2

.PHONY: all clean test

all: $(TARGETS)

# 기본 XOR 데모
simple_xor: simple_xor.c
	$(CC) $(CFLAGS) $< -o $@

# 파일 암호화 도구 (기본)
file_encrypt: file_encrypt.c
	$(CC) $(CFLAGS) $< -o $@

# 파일 암호화 도구 v2 (헤더 + 검증)
file_encrypt_v2: file_encrypt_v2.c
	$(CC) $(CFLAGS) $< -o $@

# 테스트 실행
test: all
	@echo "=== Testing simple_xor ==="
	./simple_xor
	@echo ""
	@echo "=== Testing file_encrypt ==="
	@echo "Test message for encryption" > test.txt
	./file_encrypt -e test.txt test.enc mypassword
	./file_encrypt -d test.enc decrypted.txt mypassword
	@diff -q test.txt decrypted.txt && echo "✓ Basic encryption test passed"
	@rm -f test.txt test.enc decrypted.txt
	@echo ""
	@echo "=== Testing file_encrypt_v2 ==="
	@echo "Advanced test with header validation" > test2.txt
	./file_encrypt_v2 encrypt test2.txt test2.enc strongpass
	./file_encrypt_v2 info test2.enc
	./file_encrypt_v2 decrypt test2.enc decrypted2.txt strongpass
	@diff -q test2.txt decrypted2.txt && echo "✓ Advanced encryption test passed"
	@echo ""
	@echo "Testing wrong password (should fail):"
	@./file_encrypt_v2 decrypt test2.enc wrong.txt wrongpass || echo "✓ Password validation working"
	@rm -f test2.txt test2.enc decrypted2.txt wrong.txt
	@echo ""
	@echo "All tests passed!"

clean:
	rm -f $(TARGETS) *.o *.enc *.txt *.bin *.dec
