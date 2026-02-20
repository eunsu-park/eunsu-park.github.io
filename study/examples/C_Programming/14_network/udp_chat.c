/**
 * UDP Chat (peer-to-peer)
 *
 * Simple UDP-based chat. Each instance can both send and receive.
 * Uses select() to monitor both stdin and the UDP socket.
 *
 * Build: make
 * Usage: ./udp_chat <local_port> <remote_ip> <remote_port>
 *   Example: Terminal 1: ./udp_chat 9001 127.0.0.1 9002
 *            Terminal 2: ./udp_chat 9002 127.0.0.1 9001
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define BUF_SIZE 1024

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <local_port> <remote_ip> <remote_port>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    int local_port = atoi(argv[1]);
    const char *remote_ip = argv[2];
    int remote_port = atoi(argv[3]);

    /* Create UDP socket */
    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    /* Bind to local port */
    struct sockaddr_in local_addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(local_port)
    };

    if (bind(sock_fd, (struct sockaddr *)&local_addr,
             sizeof(local_addr)) < 0) {
        perror("bind");
        close(sock_fd);
        exit(EXIT_FAILURE);
    }

    /* Setup remote address */
    struct sockaddr_in remote_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(remote_port)
    };
    inet_pton(AF_INET, remote_ip, &remote_addr.sin_addr);

    printf("UDP Chat - listening on port %d, sending to %s:%d\n",
           local_port, remote_ip, remote_port);
    printf("Type messages (Ctrl+D to quit):\n\n");

    fd_set read_fds;
    char buffer[BUF_SIZE];
    int max_fd = (sock_fd > STDIN_FILENO) ? sock_fd : STDIN_FILENO;

    while (1) {
        FD_ZERO(&read_fds);
        FD_SET(STDIN_FILENO, &read_fds);
        FD_SET(sock_fd, &read_fds);

        int ready = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (ready < 0) {
            perror("select");
            break;
        }

        /* User typed something */
        if (FD_ISSET(STDIN_FILENO, &read_fds)) {
            if (fgets(buffer, BUF_SIZE, stdin) == NULL)
                break;

            sendto(sock_fd, buffer, strlen(buffer), 0,
                   (struct sockaddr *)&remote_addr, sizeof(remote_addr));
        }

        /* Received a message */
        if (FD_ISSET(sock_fd, &read_fds)) {
            struct sockaddr_in from_addr;
            socklen_t from_len = sizeof(from_addr);
            ssize_t bytes = recvfrom(sock_fd, buffer, BUF_SIZE - 1, 0,
                                     (struct sockaddr *)&from_addr,
                                     &from_len);
            if (bytes > 0) {
                buffer[bytes] = '\0';
                char ip[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &from_addr.sin_addr, ip, sizeof(ip));
                printf("[%s:%d] %s", ip, ntohs(from_addr.sin_port), buffer);
            }
        }
    }

    close(sock_fd);
    printf("Chat ended\n");
    return 0;
}
