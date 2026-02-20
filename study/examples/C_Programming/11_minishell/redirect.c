// redirect.c
// 입출력 리다이렉션 구현
// > (출력), >> (추가), < (입력) 연산자 처리
// 컴파일: gcc -c redirect.c 또는 다른 파일과 함께 링크

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

typedef struct {
    char* input_file;   // < 파일
    char* output_file;  // > 또는 >> 파일
    int append;         // >> 인 경우 1
} Redirect;

// 리다이렉션 파싱
// args에서 리다이렉션 제거하고 Redirect 구조체에 저장
void parse_redirect(char** args, Redirect* redir) {
    redir->input_file = NULL;
    redir->output_file = NULL;
    redir->append = 0;

    int i = 0;
    int j = 0;

    while (args[i] != NULL) {
        if (strcmp(args[i], "<") == 0) {
            // 입력 리다이렉션
            if (args[i + 1]) {
                redir->input_file = args[i + 1];
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">") == 0) {
            // 출력 리다이렉션 (덮어쓰기)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 0;
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">>") == 0) {
            // 출력 리다이렉션 (추가)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 1;
                i += 2;
                continue;
            }
        }

        // 리다이렉션이 아닌 인자
        args[j++] = args[i++];
    }
    args[j] = NULL;
}

// 리다이렉션 적용 (자식 프로세스에서 호출)
int apply_redirect(Redirect* redir) {
    // 입력 리다이렉션
    if (redir->input_file) {
        int fd = open(redir->input_file, O_RDONLY);
        if (fd < 0) {
            perror(redir->input_file);
            return -1;
        }
        dup2(fd, STDIN_FILENO);
        close(fd);
    }

    // 출력 리다이렉션
    if (redir->output_file) {
        int flags = O_WRONLY | O_CREAT;
        flags |= redir->append ? O_APPEND : O_TRUNC;

        int fd = open(redir->output_file, flags, 0644);
        if (fd < 0) {
            perror(redir->output_file);
            return -1;
        }
        dup2(fd, STDOUT_FILENO);
        close(fd);
    }

    return 0;
}

// 리다이렉션을 포함한 명령어 실행
void execute_with_redirect(char** args) {
    Redirect redir;
    parse_redirect(args, &redir);

    if (args[0] == NULL) return;

    pid_t pid = fork();

    if (pid == 0) {
        // 자식: 리다이렉션 적용 후 실행
        if (apply_redirect(&redir) < 0) {
            exit(EXIT_FAILURE);
        }
        execvp(args[0], args);
        perror(args[0]);
        exit(EXIT_FAILURE);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
    } else {
        perror("fork");
    }
}

// 테스트용 메인 함수
#ifdef TEST_REDIRECT
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

    printf("=== 리다이렉션 테스트 ===\n");
    printf("예제 명령어:\n");
    printf("  ls -l > output.txt\n");
    printf("  cat < input.txt\n");
    printf("  echo \"Hello\" >> output.txt\n");
    printf("  wc -l < /etc/passwd\n");
    printf("\n종료: exit 또는 Ctrl+D\n\n");

    while (1) {
        printf("redirect> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;
        }

        if (input[0] == '\n') continue;

        // 입력 복사 (strtok이 원본 수정)
        char input_copy[MAX_INPUT];
        strncpy(input_copy, input, sizeof(input_copy));

        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "exit") == 0) {
            printf("종료합니다.\n");
            break;
        }

        // 리다이렉션 포함 실행
        execute_with_redirect(args);
    }

    return 0;
}
#endif
