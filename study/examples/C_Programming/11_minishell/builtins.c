// builtins.c
// 쉘 내장 명령어 구현
// cd, pwd, echo, help, export, env 등
// 컴파일: gcc -c builtins.c 또는 다른 파일과 함께 링크

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// 내장 명령어 이름들
const char* builtin_names[] = {
    "cd",
    "pwd",
    "echo",
    "exit",
    "help",
    "export",
    "env",
    NULL
};

// cd: 디렉토리 변경
int builtin_cd(char** args) {
    const char* path;

    if (args[1] == NULL) {
        // 인자 없으면 홈 디렉토리
        path = getenv("HOME");
        if (path == NULL) {
            fprintf(stderr, "cd: HOME 환경변수가 설정되지 않음\n");
            return 1;
        }
    } else if (strcmp(args[1], "-") == 0) {
        // cd - : 이전 디렉토리
        path = getenv("OLDPWD");
        if (path == NULL) {
            fprintf(stderr, "cd: OLDPWD 환경변수가 설정되지 않음\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(args[1], "~") == 0) {
        path = getenv("HOME");
    } else {
        path = args[1];
    }

    // 현재 디렉토리 저장
    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    // OLDPWD, PWD 환경변수 갱신
    setenv("OLDPWD", oldpwd, 1);

    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

// pwd: 현재 디렉토리 출력
int builtin_pwd(char** args) {
    (void)args;  // 사용하지 않음

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

// echo: 인자 출력
int builtin_echo(char** args) {
    int newline = 1;
    int start = 1;

    // -n 옵션: 줄바꿈 없이 출력
    if (args[1] && strcmp(args[1], "-n") == 0) {
        newline = 0;
        start = 2;
    }

    for (int i = start; args[i]; i++) {
        printf("%s", args[i]);
        if (args[i + 1]) printf(" ");
    }

    if (newline) printf("\n");
    return 0;
}

// help: 도움말
int builtin_help(char** args) {
    (void)args;

    printf("\n=== Mini Shell 도움말 ===\n\n");
    printf("내장 명령어:\n");
    printf("  cd [디렉토리]  - 디렉토리 변경\n");
    printf("  pwd           - 현재 디렉토리 출력\n");
    printf("  echo [텍스트]  - 텍스트 출력\n");
    printf("  export VAR=값  - 환경변수 설정\n");
    printf("  env           - 환경변수 목록\n");
    printf("  help          - 이 도움말\n");
    printf("  exit          - 쉘 종료\n");
    printf("\n외부 명령어는 PATH에서 검색됩니다.\n\n");

    return 0;
}

// export: 환경변수 설정
int builtin_export(char** args) {
    if (args[1] == NULL) {
        // 인자 없으면 환경변수 목록 출력
        extern char** environ;
        for (char** env = environ; *env; env++) {
            printf("export %s\n", *env);
        }
        return 0;
    }

    // VAR=value 형식 파싱
    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
            *eq = '=';
        } else {
            // = 없으면 빈 값으로 설정
            setenv(args[i], "", 1);
        }
    }

    return 0;
}

// env: 환경변수 출력
int builtin_env(char** args) {
    (void)args;

    extern char** environ;
    for (char** env = environ; *env; env++) {
        printf("%s\n", *env);
    }
    return 0;
}

// 내장 명령어인지 확인하고 실행
// 반환: -1 (내장 명령어 아님), 0+ (실행 결과)
int execute_builtin(char** args) {
    if (args[0] == NULL) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd(args);
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "help") == 0) return builtin_help(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "env") == 0) return builtin_env(args);

    return -1;  // 내장 명령어 아님
}

// 테스트용 메인 함수 (독립 실행 가능)
#ifdef TEST_BUILTINS
int main(void) {
    char input[1024];
    char* args[64];

    printf("=== 내장 명령어 테스트 ===\n");
    printf("테스트 명령어: cd, pwd, echo, help, export, env, exit\n\n");

    while (1) {
        printf("builtin> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }

        // 파싱
        int argc = 0;
        char* token = strtok(input, " \t\n");
        while (token && argc < 63) {
            args[argc++] = token;
            token = strtok(NULL, " \t\n");
        }
        args[argc] = NULL;

        if (argc == 0) continue;

        // exit 체크
        if (strcmp(args[0], "exit") == 0) {
            printf("종료합니다.\n");
            break;
        }

        // 내장 명령어 실행
        int result = execute_builtin(args);
        if (result == -1) {
            printf("알 수 없는 명령어: %s\n", args[0]);
        }
    }

    return 0;
}
#endif
