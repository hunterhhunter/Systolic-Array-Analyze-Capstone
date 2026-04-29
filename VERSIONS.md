# Pinned versions (Phase 3 onwards)

이 파일은 재현성을 위해 toolchain commit / 패키지 버전 / 호스트 환경을 고정한다.
값이 바뀌면 한 줄 짜리 이유 주석을 함께 업데이트한다.

## Toolchain commits

### IREE (로컬 빌드)
- Source repo: `/home/swlab-youngjin/iree`
- Build dir: `/home/swlab-youngjin/iree-build`
- Commit: `662a6330158c9bb39bf455ab5411444ca10f5b98`
- Commit date: 2026-03-21 11:30:51 +0000
- Binary: `/home/swlab-youngjin/iree-build/tools/iree-opt`
- Python binding (local build): `/home/swlab-youngjin/iree-build/compiler/bindings/python/`

### SCALE-Sim (vendored fork)
- Path: `SCALE-Sim/`
- Commit: `9f98c4371055a54c75209c2e02b640b897550532`
- Install: `pip install -e ./SCALE-Sim`
- **Local patch (Phase 3, 2026-04-27)**:
  `scalesim/memory/double_buffered_scratchpad_mem.py:307` 한 줄을
  `int(max(ofmap_serviced_cycles))` → `int(np.asarray(ofmap_serviced_cycles).max())`
  로 수정. numpy 2.x에서 list-of-arrays에 대한 builtin `max()`가
  ambiguous truth value로 회귀하는 것을 우회. 학부 capstone 범위 내 최소 fix
  (다른 SCALE-Sim 기능에는 영향 없음).

### iree-compiler / iree-runtime (PyPI, .venv)
- Version: `20241104.1068` (PyPI dev snapshot, 2024-11-04)
- **Drift 경고**: 로컬 IREE 빌드(commit `662a633`, 2026-03-21)보다 **약 1년 5개월 오래됨**.
  PyPI dev 채널이 주기적으로만 갱신되기 때문. Phase 3 [tools/mlir2scalesim.py](tools/mlir2scalesim.py)
  구현 시 `iree.compiler.ir.Module.parse()` + `walk()`가 로컬 `iree-opt`로 생성한 tiled MLIR을
  올바르게 파싱하는지 golden(walking_skeleton 128 tile)으로 재검증 필요. 실패 시 fallback:
  (a) 로컬 빌드 Python binding을 `PYTHONPATH=/home/swlab-youngjin/iree-build/compiler/bindings/python/`
  으로 오버라이드, (b) regex 파서로 하강.

## Host environment

- Python: 3.12.3 (venv: `.venv/`)
- OS: Ubuntu 22.04.5 LTS (kernel 6.8.0-110)
- Shell: bash

## This repo

- Git commit: `N/A (not yet initialized — Phase 8 공개 전 git init 예정)`

## 재현 절차 (clean machine 기준)

`uv` (>=0.10)를 설치해둔다. `make env`가 내부에서 `uv venv` + `uv pip install`을 호출.

1. `git clone <this-repo> npu-capston && cd npu-capston`
2. IREE 로컬 빌드 (commit `662a633...` 체크아웃, 별도 IREE README)
3. SCALE-Sim submodule 동기화 (commit `9f98c43...` 고정)
4. `make env` — `.venv/` 생성 + 런타임/개발 의존성 + `SCALE-Sim` editable install
5. `make test` (Phase 3 후반에 활성)
