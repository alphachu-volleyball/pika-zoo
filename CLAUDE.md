# pika-zoo - Claude Development Guide

## 프로젝트 개요

피카츄배구(1997) 리버스 엔지니어링 기반의 물리엔진 Python 포팅 + PettingZoo/Gymnasium 강화학습 환경.

### 목표

- 원작 JS 구현체([gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball))의 게임 물리를 Python으로 정확히 포팅
- PettingZoo `ParallelEnv` (2인 대전) + Gymnasium 래퍼로 표준 RL 인터페이스 제공
- training-center에서 Git tag dependency로 참조하여 강화학습 훈련에 사용

### 관측/행동 공간

- **관측**: 저차원 벡터 (위치, 속도 등) — GPU보다 CPU 병렬화가 병목
- **행동**: 이산 행동 공간 (방향키 + 점프 조합)

## 아키텍처

```
alphachu-volleyball/
├── pika-zoo (이 repo)        ← RL 환경 + 물리엔진
├── training-center           ← PPO, Self-play, PFSP 훈련
├── world-tournament          ← 웹 데모 (GitHub Pages)
└── vs-recorder               ← 리플레이 분석 (추후)
```

training-center → pika-zoo: Git tag pinning (`pika-zoo @ git+...@v0.1.0`)
training-center → world-tournament: ONNX 모델 (GitHub Releases)

## 개발 환경

- **Python**: 3.10+
- **패키지 관리**: uv (`pyproject.toml` + `uv.lock`)
- **Linter/Formatter**: ruff
- **테스트**: pytest

### 명령어

```bash
uv sync                  # 의존성 설치
uv run ruff check .      # 린트
uv run ruff format .     # 포맷팅
uv run pytest            # 테스트
```

## 코드 품질

### ruff 설정

```toml
[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

## 버전 관리 및 Git

### Semantic Versioning

`MAJOR.MINOR.PATCH` — Git tag로 버전 표기 (예: `v0.1.0`)

### Branch Workflow

```
feat/* ──(squash merge)──► release/{version} ──(merge commit)──► main ──► tag
fix/*  ──(squash merge)──►
```

- feat/fix → release: squash merge (PR 필수)
- release → main: merge commit (PR 필수)

### 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 형식:

```
<type>(<scope>): <subject>

feat(env): add random ball position mode
fix(physics): correct ball-net collision
docs(readme): update architecture diagram
chore(ci): add ruff lint workflow
```

주요 type: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

## CI/CD (GitHub Actions)

| 트리거 | 내용 |
|--------|------|
| PR, push to main | ruff lint, pytest |
| tag push (`v*`) | release 생성 |

학습은 CI에서 실행하지 않음 (GPU 필요, 장시간 소요).

## 코드 복사 방침

서브모듈 사용하지 않음 — 상당한 커스터마이징 필요. 외부 코드 복사 시 반드시:

- 원본 출처 URL
- 라이선스 파일 (LICENSE)
- 변경사항 기록 (ATTRIBUTION.md)

### 참고 소스

| 소스 | 라이선스 | 비고 |
|------|----------|------|
| [helpingstar/pika-zoo](https://github.com/helpingstar/pika-zoo) | MIT | PettingZoo 환경 참고 |
| [hankluo6/gym-pikachu-volleyball](https://github.com/hankluo6/gym-pikachu-volleyball) | 확인 필요 | Gymnasium 환경 참고 |
| [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) | UNLICENSED (확인 필요) | 원작 리버스 엔지니어링 JS |

## 하드웨어 참고

- AMD Ryzen 7 3700X (8C/16T), NVIDIA RTX 2080 Super (8GB)
- 저차원 벡터 관측 + MLP 정책 → CPU(환경 병렬화)가 병목
- SB3 `SubprocVecEnv`로 8~16개 환경 병렬 실행 가능
