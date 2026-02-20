# Makefile for calculator

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
TARGET = calculator

all: $(TARGET)

$(TARGET): calculator.c
	$(CC) $(CFLAGS) calculator.c -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
