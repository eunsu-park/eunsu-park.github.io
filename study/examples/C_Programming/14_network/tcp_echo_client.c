/**
 * TCP Echo Client
 *
 * Connects to a TCP echo server and sends user input.
 * Receives and displays the echoed response.
 *
 * Build: make
 * Usage: ./tcp_echo_client [server_ip] [port]
 *   Default: 127.0.0.1:8080
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define DEFAULT_PORT 8080
#define BUF_SIZE     1024

int main(int argc, char *argv[]) {
    const char *server_ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : DEFAULT_PORT;

    /* Create socket */
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    /* Connect */
    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port)
    };

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid address: %s\n", server_ip);
        close(sock_fd);
        exit(EXIT_FAILURE);
    }

    if (connect(sock_fd, (struct sockaddr *)&server_addr,
                sizeof(server_addr)) < 0) {
        perror("connect");
        close(sock_fd);
        exit(EXIT_FAILURE);
    }

    printf("Connected to %s:%d\n", server_ip, port);
    printf("Type messages (Ctrl+D to quit):\n");

    char buffer[BUF_SIZE];
    while (fgets(buffer, BUF_SIZE, stdin) != NULL) {
        /* Send message */
        ssize_t sent = send(sock_fd, buffer, strlen(buffer), 0);
        if (sent < 0) {
            perror("send");
            break;
        }

        /* Receive echo */
        ssize_t bytes = recv(sock_fd, buffer, BUF_SIZE - 1, 0);
        if (bytes <= 0) {
            printf("Server disconnected\n");
            break;
        }
        buffer[bytes] = '\0';
        printf("Echo: %s", buffer);
    }

    close(sock_fd);
    printf("Disconnected\n");
    return 0;
}
