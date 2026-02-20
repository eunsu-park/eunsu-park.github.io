// minishell_v1.c
// 기본 쉘 구조: 읽기 → 파싱 → 실행 → 반복
// 컴파일: gcc -o minishell_v1 minishell_v1.c -Wall
// 실행: ./minishell_v1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_INPUT 1024
#define MAX_ARGS 64

// 입력을 공백으로 분리
int parse_input(char* input, char** args) {
    int argc = 0;
    char* token = strtok(input, " \t\n");

    while (token != NULL && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;

    return argc;
}

// 명령어 실행
void execute(char** args) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork 실패");
        return;
    }

    if (pid == 0) {
        // 자식 프로세스: 명령어 실행
        execvp(args[0], args);
        // execvp 실패시
        perror(args[0]);
        exit(EXIT_FAILURE);
    } else {
        // 부모 프로세스: 자식 종료 대기
        int status;
        waitpid(pid, &status, 0);
    }
}

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    printf("\n=== Mini Shell v1 ===\n");
    printf("기본 명령어 실행 쉘\n");
    printf("종료: exit 명령어 또는 Ctrl+D\n\n");

    while (1) {
        // 프롬프트 출력
        printf("minish> ");
        fflush(stdout);

        // 입력 읽기
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;  // EOF (Ctrl+D)
        }

        // 빈 입력 무시
        if (input[0] == '\n') continue;

        // 파싱
        int argc = parse_input(input, args);
        if (argc == 0) continue;

        // exit 명령어
        if (strcmp(args[0], "exit") == 0) {
            printf("쉘을 종료합니다.\n");
            break;
        }

        // 실행
        execute(args);
    }

    return 0;
}
