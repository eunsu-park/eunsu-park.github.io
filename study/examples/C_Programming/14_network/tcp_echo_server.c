/**
 * TCP Echo Server with select() multiplexing
 *
 * Handles multiple clients concurrently using select().
 * Each message received from a client is echoed back.
 *
 * Build: make
 * Usage: ./tcp_echo_server [port]
 *   Default port: 8080
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

#define DEFAULT_PORT 8080
#define MAX_CLIENTS  10
#define BUF_SIZE     1024

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : DEFAULT_PORT;

    /* Create server socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    /* Allow address reuse */
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    /* Bind */
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port)
    };

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    /* Listen */
    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    printf("TCP Echo Server listening on port %d (max %d clients)\n",
           port, MAX_CLIENTS);

    /* Setup select */
    int client_fds[MAX_CLIENTS];
    for (int i = 0; i < MAX_CLIENTS; i++)
        client_fds[i] = -1;

    fd_set active_fds, read_fds;
    FD_ZERO(&active_fds);
    FD_SET(server_fd, &active_fds);
    int max_fd = server_fd;

    char buffer[BUF_SIZE];

    /* Main loop */
    while (1) {
        read_fds = active_fds;
        int ready = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("select");
            break;
        }

        /* New connection? */
        if (FD_ISSET(server_fd, &read_fds)) {
            struct sockaddr_in client_addr;
            socklen_t len = sizeof(client_addr);
            int new_fd = accept(server_fd,
                               (struct sockaddr *)&client_addr, &len);
            if (new_fd >= 0) {
                int added = 0;
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (client_fds[i] == -1) {
                        client_fds[i] = new_fd;
                        FD_SET(new_fd, &active_fds);
                        if (new_fd > max_fd) max_fd = new_fd;

                        char ip[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &client_addr.sin_addr,
                                  ip, sizeof(ip));
                        printf("[+] Client connected: %s:%d (fd=%d)\n",
                               ip, ntohs(client_addr.sin_port), new_fd);
                        added = 1;
                        break;
                    }
                }
                if (!added) {
                    const char *msg = "Server full\n";
                    send(new_fd, msg, strlen(msg), 0);
                    close(new_fd);
                }
            }
        }

        /* Check client sockets */
        for (int i = 0; i < MAX_CLIENTS; i++) {
            int fd = client_fds[i];
            if (fd == -1) continue;

            if (FD_ISSET(fd, &read_fds)) {
                ssize_t bytes = recv(fd, buffer, BUF_SIZE - 1, 0);
                if (bytes <= 0) {
                    printf("[-] Client disconnected (fd=%d)\n", fd);
                    close(fd);
                    FD_CLR(fd, &active_fds);
                    client_fds[i] = -1;
                } else {
                    buffer[bytes] = '\0';
                    printf("[fd=%d] %s", fd, buffer);
                    send(fd, buffer, bytes, 0);
                }
            }
        }
    }

    close(server_fd);
    return 0;
}
