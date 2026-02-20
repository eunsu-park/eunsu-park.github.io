/**
 * Pipe Demo: Parent-child bidirectional communication
 *
 * Demonstrates two-pipe pattern for bidirectional IPC.
 * Parent sends a task, child processes it and replies.
 *
 * Build: make
 * Usage: ./pipe_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/wait.h>

#define BUF_SIZE 256

/* Convert string to uppercase (child's task) */
static void to_upper(char *str) {
    for (int i = 0; str[i]; i++)
        str[i] = toupper((unsigned char)str[i]);
}

int main(void) {
    int parent_to_child[2];  /* Parent writes, child reads */
    int child_to_parent[2];  /* Child writes, parent reads */

    if (pipe(parent_to_child) < 0 || pipe(child_to_parent) < 0) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        /* === Child process === */
        close(parent_to_child[1]);  /* Close write end */
        close(child_to_parent[0]);  /* Close read end */

        char buffer[BUF_SIZE];
        ssize_t n;

        while ((n = read(parent_to_child[0], buffer, BUF_SIZE - 1)) > 0) {
            buffer[n] = '\0';
            printf("[Child PID=%d] Received: \"%s\"\n", getpid(), buffer);

            to_upper(buffer);
            write(child_to_parent[1], buffer, strlen(buffer));
        }

        close(parent_to_child[0]);
        close(child_to_parent[1]);
        exit(EXIT_SUCCESS);
    }

    /* === Parent process === */
    close(parent_to_child[0]);  /* Close read end */
    close(child_to_parent[1]);  /* Close write end */

    const char *messages[] = {
        "hello world",
        "pipe communication",
        "inter-process messaging"
    };

    char reply[BUF_SIZE];

    for (int i = 0; i < 3; i++) {
        printf("[Parent] Sending: \"%s\"\n", messages[i]);
        write(parent_to_child[1], messages[i], strlen(messages[i]));

        /* Small delay to let child process */
        usleep(50000);

        ssize_t n = read(child_to_parent[0], reply, BUF_SIZE - 1);
        if (n > 0) {
            reply[n] = '\0';
            printf("[Parent] Reply:   \"%s\"\n\n", reply);
        }
    }

    close(parent_to_child[1]);
    close(child_to_parent[0]);
    wait(NULL);

    printf("Done!\n");
    return 0;
}
