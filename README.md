# 개인 이력서 웹사이트

Jekyll 기반 이력서 웹사이트입니다. GitHub Pages를 통해 자동으로 호스팅됩니다.

**라이브 사이트**: [www.eunsu.me](https://www.eunsu.me)

## 빠른 시작: 내용 수정하기

각 페이지의 내용은 `_data/` 폴더의 YAML 파일에서 관리됩니다:

| 파일 | 설명 |
|------|------|
| `_data/data.yml` | 메인 CV (프로필, 학력, 경력 등) |
| `_data/publications.yml` | 전체 논문 목록 |
| `_data/portfolio.yml` | 연구 자료/포트폴리오 |
| `_data/share.yml` | 공유 파일 목록 |

GitHub에서 바로 수정하거나, 로컬에서 수정 후 push하면 자동으로 반영됩니다.

## 페이지 구조

| URL | 설명 |
|-----|------|
| `/` | 메인 CV 페이지 |
| `/publications` | 전체 논문 목록 |
| `/portfolio` | 연구 자료/포트폴리오 |
| `/share` | 외부 공유용 파일 |
| `/print` | 인쇄용 페이지 |

## 메인 CV 수정 항목 (`_data/data.yml`)

| 섹션 | 설명 |
|------|------|
| `sidebar` | 프로필 사진, 연락처, 언어, 관심분야 |
| `career-profile` | 간략한 자기소개 |
| `education` | 학력 |
| `experiences` | 경력 |
| `publications` | 논문 (제1저자, 요약) |
| `projects` | 연구 과제 (PI) |
| `patents` | 특허 |

## 프로필 사진 변경

1. 새 이미지를 `assets/images/` 폴더에 저장
2. `_data/data.yml`의 `avatar` 값을 새 파일명으로 변경

## 테마 색상 변경

`_config.yml` 파일에서 `theme_skin` 값 변경:

```yaml
theme_skin: ceramic   # 선택: blue, turquoise, green, berry, orange, ceramic
```

## 로컬에서 미리보기 (선택사항)

변경사항을 GitHub에 올리기 전에 로컬에서 확인하고 싶다면:

```bash
# Ruby가 설치되어 있어야 합니다
bundle install
bundle exec jekyll serve
```

브라우저에서 `http://localhost:4000` 접속

## 폴더 구조

```
├── _data/
│   ├── data.yml           # ⭐ 메인 CV 내용
│   ├── publications.yml   # 논문 목록
│   ├── portfolio.yml      # 포트폴리오 항목
│   └── share.yml          # 공유 파일 목록
├── _config.yml            # 사이트 설정 (테마, URL 등)
├── _includes/             # HTML 템플릿 조각
├── _layouts/              # 페이지 레이아웃
├── _sass/                 # 스타일시트
├── assets/                # 이미지, CSS, JS
├── _archive/              # 미사용 파일 보관
├── index.html             # 메인 페이지
├── publications.html      # 논문 페이지
├── portfolio.html         # 포트폴리오 페이지
├── share.html             # 공유 페이지
└── print.html             # 인쇄용 페이지
```

## 원본 테마

이 사이트는 [Orbit](https://github.com/sharu725/online-cv) 테마를 기반으로 합니다.
- 디자인: [Xiaoying Riley](http://themes.3rdwavemedia.com/)
- Jekyll 포팅: [sharu725](https://github.com/sharu725)
