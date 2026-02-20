// pipe.c
// 파이프 구현
// 여러 명령어를 | 로 연결하여 실행
// 컴파일: gcc -c pipe.c 또는 다른 파일과 함께 링크

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_PIPES 10

// 파이프로 분리된 명령어 수 카운트
int count_pipes(char** args) {
    int count = 0;
    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) count++;
    }
    return count;
}

// 파이프 위치에서 args 분리
// commands[0] = 첫 번째 명령어의 args
// commands[1] = 두 번째 명령어의 args
// ...
int split_by_pipe(char** args, char*** commands) {
    int cmd_count = 0;
    commands[cmd_count++] = args;

    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) {
            args[i] = NULL;  // 파이프 위치를 NULL로
            if (args[i + 1]) {
                commands[cmd_count++] = &args[i + 1];
            }
        }
    }

    return cmd_count;
}

// 파이프 실행
void execute_pipe(char** args) {
    char** commands[MAX_PIPES + 1];
    int cmd_count = split_by_pipe(args, commands);

    if (cmd_count == 1) {
        // 파이프 없음: 일반 실행
        pid_t pid = fork();
        if (pid == 0) {
            execvp(commands[0][0], commands[0]);
            perror(commands[0][0]);
            exit(EXIT_FAILURE);
        } else if (pid > 0) {
            wait(NULL);
        } else {
            perror("fork");
        }
        return;
    }

    int pipes[MAX_PIPES][2];  // 파이프 파일 디스크립터

    // 파이프 생성
    for (int i = 0; i < cmd_count - 1; i++) {
        if (pipe(pipes[i]) < 0) {
            perror("pipe");
            return;
        }
    }

    // 각 명령어 실행
    for (int i = 0; i < cmd_count; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // 자식 프로세스

            // 이전 파이프의 읽기 끝을 stdin으로
            if (i > 0) {
                dup2(pipes[i - 1][0], STDIN_FILENO);
            }

            // 다음 파이프의 쓰기 끝을 stdout으로
            if (i < cmd_count - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // 모든 파이프 닫기
            for (int j = 0; j < cmd_count - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // 명령어 실행
            execvp(commands[i][0], commands[i]);
            perror(commands[i][0]);
            exit(EXIT_FAILURE);

        } else if (pid < 0) {
            perror("fork");
            return;
        }
    }

    // 부모: 모든 파이프 닫기
    for (int i = 0; i < cmd_count - 1; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    // 모든 자식 프로세스 대기
    for (int i = 0; i < cmd_count; i++) {
        wait(NULL);
    }
}

// 테스트용 메인 함수
#ifdef TEST_PIPE
#define MAX_INPUT 1024
#define MAX_ARGS 64

int parse_args(char* input, char** args) {
    int argc = 0;
    char* token = strtok(input, " \t\n");
    while (token && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;
    return argc;
}

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    printf("=== 파이프 테스트 ===\n");
    printf("예제 명령어:\n");
    printf("  ls -l | grep \".c\"\n");
    printf("  cat /etc/passwd | wc -l\n");
    printf("  ps aux | grep bash | head -5\n");
    printf("  ls | sort | uniq\n");
    printf("\n종료: exit 또는 Ctrl+D\n\n");

    while (1) {
        printf("pipe> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;
        }

        if (input[0] == '\n') continue;

        // 입력 복사
        char input_copy[MAX_INPUT];
        strncpy(input_copy, input, sizeof(input_copy));

        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "exit") == 0) {
            printf("종료합니다.\n");
            break;
        }

        // 파이프 실행
        execute_pipe(args);
    }

    return 0;
}
#endif
