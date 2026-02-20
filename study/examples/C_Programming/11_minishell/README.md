# Mini Shell 예제

간단한 명령어 쉘 구현 예제입니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `minishell_v1.c` | 기본 쉘 구조 (명령어 파싱 및 실행) |
| `builtins.c` | 내장 명령어 구현 (cd, pwd, echo, help 등) |
| `redirect.c` | 입출력 리다이렉션 (>, >>, <) |
| `pipe.c` | 파이프 구현 (\|) |
| `minishell.c` | 완성된 미니 쉘 (모든 기능 통합) |

## 컴파일 및 실행

### 1. 기본 쉘 (minishell_v1)

가장 간단한 쉘 구조입니다.

```bash
gcc -o minishell_v1 minishell_v1.c -Wall
./minishell_v1
```

**기능:**
- 외부 명령어 실행 (fork + exec)
- exit 명령어

**예제:**
```bash
minish> ls -l
minish> pwd
minish> echo hello world
minish> exit
```

### 2. 내장 명령어 테스트 (builtins)

내장 명령어만 따로 테스트합니다.

```bash
gcc -o builtins builtins.c -Wall -DTEST_BUILTINS
./builtins
```

**기능:**
- cd (디렉토리 변경)
- pwd (현재 디렉토리)
- echo (텍스트 출력)
- export (환경변수 설정)
- env (환경변수 목록)
- help (도움말)

**예제:**
```bash
builtin> pwd
builtin> cd /tmp
builtin> pwd
builtin> cd -
builtin> export MY_VAR=hello
builtin> echo $MY_VAR
builtin> exit
```

### 3. 리다이렉션 테스트 (redirect)

입출력 리다이렉션을 테스트합니다.

```bash
gcc -o redirect redirect.c -Wall -DTEST_REDIRECT
./redirect
```

**기능:**
- `>` : 출력을 파일로 (덮어쓰기)
- `>>` : 출력을 파일에 추가
- `<` : 파일에서 입력

**예제:**
```bash
redirect> ls -l > files.txt
redirect> cat < files.txt
redirect> echo "추가 내용" >> files.txt
redirect> wc -l < files.txt
redirect> exit
```

### 4. 파이프 테스트 (pipe)

파이프 기능을 테스트합니다.

```bash
gcc -o pipe pipe.c -Wall -DTEST_PIPE
./pipe
```

**기능:**
- `|` : 명령어 출력을 다음 명령어 입력으로

**예제:**
```bash
pipe> ls -l | grep ".c"
pipe> cat /etc/passwd | wc -l
pipe> ps aux | grep bash | head -5
pipe> exit
```

### 5. 완성된 미니 쉘 (minishell)

모든 기능이 통합된 완성본입니다.

```bash
gcc -o minishell minishell.c -Wall -Wextra
./minishell
```

**기능:**
- 내장 명령어 (cd, pwd, echo, export, unset, help, exit)
- 입출력 리다이렉션 (>, >>, <)
- 파이프 (|)
- 환경변수 확장 ($VAR)
- 시그널 처리 (Ctrl+C)
- 컬러 프롬프트
- 종료 코드 표시

**예제:**
```bash
~ ❯ help
~ ❯ pwd
/Users/username
~ ❯ cd /tmp
/tmp ❯ ls -la
/tmp ❯ echo $HOME
/Users/username
/tmp ❯ export MY_VAR=hello
/tmp ❯ echo $MY_VAR
hello
/tmp ❯ ls -l | grep ".txt" | wc -l
/tmp ❯ cat /etc/passwd | head -5 > first5.txt
/tmp ❯ cat first5.txt
/tmp ❯ cd -
/Users/username
~ ❯ exit
```

## Makefile 사용

모든 파일을 한 번에 컴파일:

```bash
make all
```

개별 실행 파일 컴파일:

```bash
make minishell_v1
make builtins
make redirect
make pipe
make minishell
```

정리:

```bash
make clean
```

## 주요 시스템 콜

| 함수 | 설명 |
|------|------|
| `fork()` | 프로세스 복제 |
| `execvp()` | 프로그램 실행 |
| `wait()` / `waitpid()` | 자식 프로세스 대기 |
| `pipe()` | 파이프 생성 |
| `dup2()` | 파일 디스크립터 복제 |
| `open()` | 파일 열기 |
| `chdir()` | 디렉토리 변경 |
| `getcwd()` | 현재 디렉토리 얻기 |
| `setenv()` / `getenv()` | 환경변수 설정/조회 |
| `signal()` | 시그널 핸들러 등록 |

## 학습 순서

1. **minishell_v1.c** - 쉘의 기본 구조 이해
2. **builtins.c** - 내장 명령어와 외부 명령어의 차이 이해
3. **redirect.c** - 파일 디스크립터와 리다이렉션 이해
4. **pipe.c** - 프로세스 간 통신 이해
5. **minishell.c** - 모든 기능을 통합한 완성본

## 추가 개선 아이디어

- [ ] 히스토리 기능 (history 명령어)
- [ ] 백그라운드 실행 (&)
- [ ] 와일드카드 확장 (*)
- [ ] 세미콜론 지원 (cmd1 ; cmd2)
- [ ] 논리 연산자 (&& 와 ||)
- [ ] 따옴표 처리 ("hello world")
- [ ] 탭 자동완성 (readline 라이브러리)
- [ ] 작업 제어 (jobs, fg, bg)

## 참고 문서

- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/12_Project_Mini_Shell.md`
