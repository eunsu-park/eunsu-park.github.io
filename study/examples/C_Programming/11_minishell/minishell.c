// minishell.c
// 완성된 미니 쉘
// 기능: 내장 명령어, 리다이렉션, 파이프, 시그널 처리
// 컴파일: gcc -o minishell minishell.c -Wall -Wextra
// 실행: ./minishell

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#define MAX_INPUT 1024
#define MAX_ARGS 64
#define MAX_PIPES 10

// ============ 전역 변수 ============
static int last_exit_status = 0;

// ============ 시그널 핸들러 ============
void sigint_handler(int sig) {
    (void)sig;
    printf("\n");
    // 프롬프트 다시 출력하지 않음 (메인 루프에서 처리)
}

// ============ 유틸리티 ============

// 문자열 양쪽 공백 제거
char* trim(char* str) {
    while (*str == ' ' || *str == '\t') str++;

    if (*str == '\0') return str;

    char* end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n')) {
        *end-- = '\0';
    }

    return str;
}

// ============ 파싱 ============

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

// ============ 리다이렉션 ============

typedef struct {
    char* infile;
    char* outfile;
    int append;
} Redirect;

void parse_redirect(char** args, Redirect* r) {
    r->infile = NULL;
    r->outfile = NULL;
    r->append = 0;

    int i = 0, j = 0;
    while (args[i]) {
        if (strcmp(args[i], "<") == 0 && args[i+1]) {
            r->infile = args[i+1];
            i += 2;
        } else if (strcmp(args[i], ">") == 0 && args[i+1]) {
            r->outfile = args[i+1];
            r->append = 0;
            i += 2;
        } else if (strcmp(args[i], ">>") == 0 && args[i+1]) {
            r->outfile = args[i+1];
            r->append = 1;
            i += 2;
        } else {
            args[j++] = args[i++];
        }
    }
    args[j] = NULL;
}

int setup_redirect(Redirect* r) {
    if (r->infile) {
        int fd = open(r->infile, O_RDONLY);
        if (fd < 0) { perror(r->infile); return -1; }
        dup2(fd, STDIN_FILENO);
        close(fd);
    }
    if (r->outfile) {
        int flags = O_WRONLY | O_CREAT | (r->append ? O_APPEND : O_TRUNC);
        int fd = open(r->outfile, flags, 0644);
        if (fd < 0) { perror(r->outfile); return -1; }
        dup2(fd, STDOUT_FILENO);
        close(fd);
    }
    return 0;
}

// ============ 내장 명령어 ============

int builtin_cd(char** args) {
    const char* path = args[1] ? args[1] : getenv("HOME");

    if (path == NULL) {
        fprintf(stderr, "cd: HOME 환경변수가 설정되지 않음\n");
        return 1;
    }

    if (strcmp(path, "-") == 0) {
        path = getenv("OLDPWD");
        if (!path) {
            fprintf(stderr, "cd: OLDPWD not set\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(path, "~") == 0) {
        path = getenv("HOME");
    }

    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    setenv("OLDPWD", oldpwd, 1);
    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

int builtin_pwd(void) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

int builtin_echo(char** args) {
    int newline = 1, start = 1;
    if (args[1] && strcmp(args[1], "-n") == 0) {
        newline = 0;
        start = 2;
    }

    for (int i = start; args[i]; i++) {
        // 환경변수 확장 ($VAR)
        if (args[i][0] == '$') {
            char* val = getenv(args[i] + 1);
            printf("%s", val ? val : "");
        } else {
            printf("%s", args[i]);
        }
        if (args[i + 1]) printf(" ");
    }
    if (newline) printf("\n");
    return 0;
}

int builtin_export(char** args) {
    if (!args[1]) {
        extern char** environ;
        for (char** e = environ; *e; e++) {
            printf("export %s\n", *e);
        }
        return 0;
    }

    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
            *eq = '=';
        }
    }
    return 0;
}

int builtin_unset(char** args) {
    for (int i = 1; args[i]; i++) {
        unsetenv(args[i]);
    }
    return 0;
}

int builtin_help(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════╗\n");
    printf("║        Mini Shell 도움말              ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 내장 명령어:                          ║\n");
    printf("║   cd [dir]    디렉토리 변경           ║\n");
    printf("║   pwd         현재 디렉토리           ║\n");
    printf("║   echo [...]  텍스트 출력             ║\n");
    printf("║   export V=X  환경변수 설정           ║\n");
    printf("║   unset VAR   환경변수 삭제           ║\n");
    printf("║   help        이 도움말               ║\n");
    printf("║   exit [N]    쉘 종료                 ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 리다이렉션:                           ║\n");
    printf("║   cmd > file  출력을 파일로           ║\n");
    printf("║   cmd >> file 출력을 파일에 추가      ║\n");
    printf("║   cmd < file  파일에서 입력           ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 파이프:                               ║\n");
    printf("║   cmd1 | cmd2 출력을 다음 명령 입력으로 ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    printf("\n");
    return 0;
}

// 내장 명령어 실행 (-1: 내장 아님)
int run_builtin(char** args) {
    if (!args[0]) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd();
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "unset") == 0) return builtin_unset(args);
    if (strcmp(args[0], "help") == 0) return builtin_help();

    return -1;
}

// ============ 파이프 실행 ============

int split_pipe(char** args, char*** cmds) {
    int n = 0;
    cmds[n++] = args;

    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) {
            args[i] = NULL;
            if (args[i + 1]) {
                cmds[n++] = &args[i + 1];
            }
        }
    }
    return n;
}

void run_pipeline(char** args) {
    char** cmds[MAX_PIPES + 1];
    int n = split_pipe(args, cmds);

    // 파이프 없으면 단일 명령 실행
    if (n == 1) {
        Redirect r;
        parse_redirect(cmds[0], &r);

        if (!cmds[0][0]) return;

        // 내장 명령어 체크
        int builtin_result = run_builtin(cmds[0]);
        if (builtin_result != -1) {
            last_exit_status = builtin_result;
            return;
        }

        // 외부 명령어
        pid_t pid = fork();
        if (pid == 0) {
            setup_redirect(&r);
            execvp(cmds[0][0], cmds[0]);
            fprintf(stderr, "%s: command not found\n", cmds[0][0]);
            exit(127);
        } else if (pid > 0) {
            int status;
            waitpid(pid, &status, 0);
            last_exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
        return;
    }

    // 파이프가 있는 경우
    int pipes[MAX_PIPES][2];
    for (int i = 0; i < n - 1; i++) {
        pipe(pipes[i]);
    }

    for (int i = 0; i < n; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // 입력 연결
            if (i > 0) {
                dup2(pipes[i-1][0], STDIN_FILENO);
            }
            // 출력 연결
            if (i < n - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // 모든 파이프 닫기
            for (int j = 0; j < n - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // 리다이렉션 처리 (첫/마지막 명령에만 적용)
            Redirect r;
            parse_redirect(cmds[i], &r);
            if (i == 0 && r.infile) {
                int fd = open(r.infile, O_RDONLY);
                if (fd >= 0) { dup2(fd, STDIN_FILENO); close(fd); }
            }
            if (i == n - 1 && r.outfile) {
                int flags = O_WRONLY | O_CREAT | (r.append ? O_APPEND : O_TRUNC);
                int fd = open(r.outfile, flags, 0644);
                if (fd >= 0) { dup2(fd, STDOUT_FILENO); close(fd); }
            }

            execvp(cmds[i][0], cmds[i]);
            fprintf(stderr, "%s: command not found\n", cmds[i][0]);
            exit(127);
        }
    }

    // 부모: 파이프 닫고 대기
    for (int i = 0; i < n - 1; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    int status;
    for (int i = 0; i < n; i++) {
        wait(&status);
    }
    last_exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

// ============ 프롬프트 ============

void print_prompt(void) {
    char cwd[256];
    char* dir = getcwd(cwd, sizeof(cwd));

    // 홈 디렉토리를 ~로 표시
    char* home = getenv("HOME");
    if (home && dir && strncmp(dir, home, strlen(home)) == 0) {
        printf("\033[1;34m~%s\033[0m", dir + strlen(home));
    } else {
        printf("\033[1;34m%s\033[0m", dir ? dir : "?");
    }

    // 종료 코드에 따라 색상 변경
    if (last_exit_status == 0) {
        printf(" \033[1;32m❯\033[0m ");
    } else {
        printf(" \033[1;31m❯\033[0m ");
    }

    fflush(stdout);
}

// ============ 메인 ============

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    // 시그널 핸들러 설정
    signal(SIGINT, sigint_handler);

    printf("\n\033[1;36m=== Mini Shell ===\033[0m\n");
    printf("'help' 입력하여 도움말 보기\n\n");

    while (1) {
        print_prompt();

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\nexit\n");
            break;
        }

        char* trimmed = trim(input);
        if (*trimmed == '\0') continue;

        // 주석 무시
        if (trimmed[0] == '#') continue;

        // 입력 복사 (strtok이 원본 수정)
        char input_copy[MAX_INPUT];
        strncpy(input_copy, trimmed, sizeof(input_copy));

        // 파싱
        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        // exit 명령어
        if (strcmp(args[0], "exit") == 0) {
            int code = args[1] ? atoi(args[1]) : last_exit_status;
            printf("exit\n");
            exit(code);
        }

        // 실행
        run_pipeline(args);
    }

    return last_exit_status;
}
