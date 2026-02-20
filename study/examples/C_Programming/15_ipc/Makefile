CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic

# Linux needs -lrt for shm_open, -lpthread for sem_open
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS_SHM = -lrt -lpthread
else
    LDFLAGS_SHM =
endif

TARGETS = pipe_demo shm_producer shm_consumer signal_handler

.PHONY: all clean

all: $(TARGETS)

pipe_demo: pipe_demo.c
	$(CC) $(CFLAGS) -o $@ $<

shm_producer: shm_producer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_SHM)

shm_consumer: shm_consumer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_SHM)

signal_handler: signal_handler.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
