CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic
TARGETS = tcp_echo_server tcp_echo_client udp_chat

.PHONY: all clean

all: $(TARGETS)

tcp_echo_server: tcp_echo_server.c
	$(CC) $(CFLAGS) -o $@ $<

tcp_echo_client: tcp_echo_client.c
	$(CC) $(CFLAGS) -o $@ $<

udp_chat: udp_chat.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
