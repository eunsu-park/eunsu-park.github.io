"""
01_lexer.py - Lexer/Scanner for a simple C-like language

Demonstrates lexical analysis: the first phase of compilation.
The lexer reads source code (a stream of characters) and produces
a stream of tokens, discarding whitespace and comments.

Topics covered:
  - Regular expressions for token patterns
  - Token types and Token dataclass
  - Handling keywords vs identifiers
  - String literal and character literal scanning
  - Line/column tracking for error messages
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(Enum):
    # Literals
    INT_LITERAL    = auto()
    FLOAT_LITERAL  = auto()
    STRING_LITERAL = auto()
    CHAR_LITERAL   = auto()
    BOOL_LITERAL   = auto()

    # Keywords
    INT    = auto()
    FLOAT  = auto()
    BOOL   = auto()
    CHAR   = auto()
    VOID   = auto()
    IF     = auto()
    ELSE   = auto()
    WHILE  = auto()
    FOR    = auto()
    RETURN = auto()
    TRUE   = auto()
    FALSE  = auto()
    NULL   = auto()

    # Identifiers
    IDENTIFIER = auto()

    # Arithmetic operators
    PLUS    = auto()   # +
    MINUS   = auto()   # -
    STAR    = auto()   # *
    SLASH   = auto()   # /
    PERCENT = auto()   # %

    # Relational operators
    EQ  = auto()   # ==
    NEQ = auto()   # !=
    LT  = auto()   # <
    GT  = auto()   # >
    LEQ = auto()   # <=
    GEQ = auto()   # >=

    # Logical operators
    AND = auto()   # &&
    OR  = auto()   # ||
    NOT = auto()   # !

    # Assignment
    ASSIGN      = auto()   # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN= auto()   # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN= auto()   # /=

    # Bitwise
    AMP   = auto()   # &
    PIPE  = auto()   # |
    CARET = auto()   # ^
    TILDE = auto()   # ~
    LSHIFT= auto()   # <<
    RSHIFT= auto()   # >>

    # Punctuation
    LPAREN    = auto()   # (
    RPAREN    = auto()   # )
    LBRACE    = auto()   # {
    RBRACE    = auto()   # }
    LBRACKET  = auto()   # [
    RBRACKET  = auto()   # ]
    SEMICOLON = auto()   # ;
    COLON     = auto()   # :
    COMMA     = auto()   # ,
    DOT       = auto()   # .
    ARROW     = auto()   # ->
    ELLIPSIS  = auto()   # ...

    # Special
    EOF     = auto()
    INVALID = auto()


KEYWORDS = {
    "int":    TokenType.INT,
    "float":  TokenType.FLOAT,
    "bool":   TokenType.BOOL,
    "char":   TokenType.CHAR,
    "void":   TokenType.VOID,
    "if":     TokenType.IF,
    "else":   TokenType.ELSE,
    "while":  TokenType.WHILE,
    "for":    TokenType.FOR,
    "return": TokenType.RETURN,
    "true":   TokenType.TRUE,
    "false":  TokenType.FALSE,
    "null":   TokenType.NULL,
}


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

# Each entry: (TokenType, regex_pattern)
# Patterns are tried in order; first match wins.
TOKEN_PATTERNS = [
    # Multi-character operators (must come before single-char)
    (TokenType.ELLIPSIS,     r'\.\.\.'),
    (TokenType.ARROW,        r'->'),
    (TokenType.EQ,           r'=='),
    (TokenType.NEQ,          r'!='),
    (TokenType.LEQ,          r'<='),
    (TokenType.GEQ,          r'>='),
    (TokenType.AND,          r'&&'),
    (TokenType.OR,           r'\|\|'),
    (TokenType.PLUS_ASSIGN,  r'\+='),
    (TokenType.MINUS_ASSIGN, r'-='),
    (TokenType.STAR_ASSIGN,  r'\*='),
    (TokenType.SLASH_ASSIGN, r'/='),
    (TokenType.LSHIFT,       r'<<'),
    (TokenType.RSHIFT,       r'>>'),

    # Float literal (must come before INT_LITERAL)
    (TokenType.FLOAT_LITERAL, r'\d+\.\d*([eE][+-]?\d+)?|\d+[eE][+-]?\d+'),

    # Int literal (hex, octal, decimal)
    (TokenType.INT_LITERAL,  r'0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*'),

    # String literal
    (TokenType.STRING_LITERAL, r'"(?:[^"\\]|\\.)*"'),

    # Char literal
    (TokenType.CHAR_LITERAL, r"'(?:[^'\\]|\\.)'"),

    # Identifier / keyword
    (TokenType.IDENTIFIER,   r'[A-Za-z_]\w*'),

    # Single-char operators and punctuation
    (TokenType.PLUS,      r'\+'),
    (TokenType.MINUS,     r'-'),
    (TokenType.STAR,      r'\*'),
    (TokenType.SLASH,     r'/'),
    (TokenType.PERCENT,   r'%'),
    (TokenType.LT,        r'<'),
    (TokenType.GT,        r'>'),
    (TokenType.NOT,       r'!'),
    (TokenType.ASSIGN,    r'='),
    (TokenType.AMP,       r'&'),
    (TokenType.PIPE,      r'\|'),
    (TokenType.CARET,     r'\^'),
    (TokenType.TILDE,     r'~'),
    (TokenType.LPAREN,    r'\('),
    (TokenType.RPAREN,    r'\)'),
    (TokenType.LBRACE,    r'\{'),
    (TokenType.RBRACE,    r'\}'),
    (TokenType.LBRACKET,  r'\['),
    (TokenType.RBRACKET,  r'\]'),
    (TokenType.SEMICOLON, r';'),
    (TokenType.COLON,     r':'),
    (TokenType.COMMA,     r','),
    (TokenType.DOT,       r'\.'),
]

# Compile all patterns into one master regex with named groups
_MASTER_PATTERN = re.compile(
    '|'.join(f'(?P<T{i}_{tt.name}>{pat})' for i, (tt, pat) in enumerate(TOKEN_PATTERNS))
)


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Lexer error at line {line}, col {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """
    Tokenizes a source string into a list of Tokens.
    Handles:
      - Single-line comments  // ...
      - Block comments        /* ... */
      - Whitespace (ignored)
      - All token types defined above
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def _skip_whitespace_and_comments(self) -> None:
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in ' \t\r':
                self.pos += 1
                self.column += 1
            elif ch == '\n':
                self.pos += 1
                self.line += 1
                self.column = 1
            elif self.source[self.pos:self.pos+2] == '//':
                # Single-line comment: skip to end of line
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self.pos += 1
            elif self.source[self.pos:self.pos+2] == '/*':
                # Block comment
                start_line, start_col = self.line, self.column
                self.pos += 2
                self.column += 2
                while self.pos < len(self.source):
                    if self.source[self.pos:self.pos+2] == '*/':
                        self.pos += 2
                        self.column += 2
                        break
                    elif self.source[self.pos] == '\n':
                        self.pos += 1
                        self.line += 1
                        self.column = 1
                    else:
                        self.pos += 1
                        self.column += 1
                else:
                    raise LexerError("Unterminated block comment", start_line, start_col)
            else:
                break

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                tokens.append(Token(TokenType.EOF, '', self.line, self.column))
                break

            m = _MASTER_PATTERN.match(self.source, self.pos)
            if not m:
                raise LexerError(
                    f"Unexpected character {self.source[self.pos]!r}",
                    self.line, self.column
                )

            value = m.group()
            tok_line, tok_col = self.line, self.column

            # Determine which pattern matched
            token_type = TokenType.INVALID
            for i, (tt, _) in enumerate(TOKEN_PATTERNS):
                group_name = f'T{i}_{tt.name}'
                if m.group(group_name) is not None:
                    token_type = tt
                    break

            # Reclassify identifiers that are keywords
            if token_type == TokenType.IDENTIFIER and value in KEYWORDS:
                token_type = KEYWORDS[value]

            tokens.append(Token(token_type, value, tok_line, tok_col))

            # Advance position and column
            newlines = value.count('\n')
            if newlines:
                self.line += newlines
                self.column = len(value) - value.rfind('\n')
            else:
                self.column += len(value)
            self.pos = m.end()

        return tokens


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_PROGRAM = r"""
// Compute the nth Fibonacci number
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    int a = 0;
    int b = 1;
    int i = 2;
    while (i <= n) {
        int tmp = a + b;
        a = b;
        b = tmp;
        i += 1;
    }
    return b;
}

/* Entry point */
int main() {
    float pi = 3.14159;
    char greeting[] = "Hello, World!";
    bool flag = true;
    int result = fibonacci(10);
    return 0;
}
"""


def main():
    print("=" * 60)
    print("Lexer Demo: Tokenizing a C-like program")
    print("=" * 60)
    print("\nSource program:")
    print(SAMPLE_PROGRAM)

    lexer = Lexer(SAMPLE_PROGRAM)
    tokens = lexer.tokenize()

    print(f"\nProduced {len(tokens)} tokens:\n")
    print(f"{'TYPE':<22} {'VALUE':<20} {'LINE':>4} {'COL':>4}")
    print("-" * 56)
    for tok in tokens:
        if tok.type == TokenType.EOF:
            break
        print(f"{tok.type.name:<22} {tok.value!r:<20} {tok.line:>4} {tok.column:>4}")

    print("\n--- Token type summary ---")
    from collections import Counter
    counts = Counter(tok.type.name for tok in tokens if tok.type != TokenType.EOF)
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<22}: {count}")


if __name__ == "__main__":
    main()
