# Web_Development 예제

Web_Development 폴더의 14개 레슨에 해당하는 실행 가능한 예제 코드입니다.

## 폴더 구조

```
examples/
├── 01_html_basics/       # HTML 기초
│   ├── index.html        # 기본 태그
│   └── semantic.html     # 시맨틱 HTML
│
├── 02_html_forms/        # 폼과 테이블
│   ├── form_example.html # 폼 요소
│   └── table_example.html # 테이블
│
├── 03_css_basics/        # CSS 기초
│   ├── index.html        # 선택자, 박스모델
│   └── style.css
│
├── 04_css_layout/        # CSS 레이아웃
│   ├── flexbox.html      # Flexbox
│   ├── grid.html         # Grid
│   └── style.css
│
├── 05_css_responsive/    # 반응형 디자인
│   ├── index.html        # 미디어쿼리
│   └── style.css
│
├── 06_js_basics/         # JavaScript 기초
│   ├── index.html        # 변수, 함수, 배열, 객체
│   └── script.js
│
├── 07_js_dom/            # DOM 조작
│   ├── index.html        # DOM API, 이벤트
│   └── script.js
│
├── 08_js_async/          # 비동기 프로그래밍
│   ├── index.html        # Promise, async/await
│   └── script.js
│
├── 09_project_todo/      # Todo 앱 프로젝트
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── 10_project_weather/   # 날씨 앱 프로젝트
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── 11_typescript/        # TypeScript
│   ├── basics.ts         # 타입 기초
│   ├── interfaces.ts     # 인터페이스
│   └── tsconfig.json
│
├── 12_accessibility/     # 웹 접근성
│   ├── index.html        # ARIA, 키보드
│   └── style.css
│
├── 13_seo/               # SEO
│   └── index.html        # 메타태그, 구조화데이터
│
└── 14_build_tools/       # 빌드 도구
    ├── vite-project/
    └── webpack-example/
```

## 실행 방법

### HTML/CSS/JS 예제

```bash
# 방법 1: 브라우저에서 직접 열기
open examples/01_html_basics/index.html

# 방법 2: VS Code Live Server
# VS Code에서 Live Server 확장 설치 후 Go Live 클릭

# 방법 3: Python 간이 서버
cd examples
python -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

### TypeScript 예제

```bash
# TypeScript 설치
npm install -g typescript

# 컴파일
cd examples/11_typescript
tsc basics.ts

# 또는 ts-node로 직접 실행
npx ts-node basics.ts
```

### 빌드 도구 예제

```bash
# Vite
cd examples/14_build_tools/vite-project
npm install
npm run dev

# Webpack
cd examples/14_build_tools/webpack-example
npm install
npm run build
npm run dev
```

## 레슨별 예제 목록

| 레슨 | 주제 | 예제 파일 |
|------|------|-----------|
| 01 | HTML 기초 | index.html, semantic.html |
| 02 | HTML 폼/테이블 | form_example.html, table_example.html |
| 03 | CSS 기초 | 선택자, 박스모델, 텍스트 |
| 04 | CSS 레이아웃 | Flexbox, Grid |
| 05 | 반응형 디자인 | 미디어쿼리, 모바일 퍼스트 |
| 06 | JS 기초 | 변수, 함수, 배열, 객체, 클래스 |
| 07 | DOM/이벤트 | DOM 조작, 이벤트 처리 |
| 08 | 비동기 JS | Promise, async/await, fetch |
| 09 | Todo 앱 | CRUD, 로컬스토리지, 필터 |
| 10 | 날씨 앱 | API 연동, 비동기 처리 |
| 11 | TypeScript | 타입, 인터페이스, 제네릭 |
| 12 | 웹 접근성 | ARIA, 키보드, 스크린리더 |
| 13 | SEO | 메타태그, 구조화 데이터 |
| 14 | 빌드 도구 | Vite, Webpack |

## 학습 순서

1. **기초**: 01 → 02 → 03 → 04 → 05
2. **JavaScript**: 06 → 07 → 08
3. **프로젝트**: 09 → 10
4. **고급**: 11 → 12 → 13 → 14

## 브라우저 호환성

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 참고 자료

- [MDN Web Docs](https://developer.mozilla.org/ko/)
- [CSS-Tricks](https://css-tricks.com/)
- [JavaScript.info](https://javascript.info/)
