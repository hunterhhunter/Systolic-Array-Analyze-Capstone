# TODOS — npu-capston

생성: 2026-04-21 (via /plan-eng-review)
갱신 규칙: 완료한 항목은 체크박스 [x] 표시하고 완료일 메모. 의미 없어진 항목은 ~취소~ 처리.

---

## Should-have (권장)

### `make env` 타겟 — 재현성 entry point
- **What**: IREE clone + 빌드 + SCALE-Sim `pip install -e .` + `pip install -r requirements.txt` + `VERSIONS.md` 커밋 해시 검증을 하나의 `make env`로 집약.
- **Why**: §7 Must-have "30분 재현" 약속의 실제 진입점. IREE 빌드 시간 자체는 줄일 수 없으나 "한 줄로 환경 준비" 보장.
- **언제**: Phase 8 준비기 (주 11-12) 작성 — 이 시점에는 IREE commit이 확정됐을 것.
- **의존**: `VERSIONS.md` 선행 (CQ4 이슈에서 결정).
- **예상 비용**: human 3h / CC 30min.

---

## Reminders (특정 Phase 시작 시 상기)

### 주 7 말 — 논문 draft 병행 시작
- **What**: Phase 6의 첫 figure가 나오는 주 7 말부터 Introduction · Background · Methodology 초안 병행 작성. Results 섹션은 Phase 6 결과 수집 후 삽입.
- **Why**: Phase 8 (주 12-13)만으로 영어 논문 초안을 쓰기에는 타이트함. 실험하면서 병행 작성하는 사이클이 3-4배 빠름.
- **트리거**: Phase 4 종료 게이트 (주 5 말)에서 이 reminder를 다시 확인.
- **의존**: Phase 6 착수.
- **예상 비용**: 병행 1h/일 × 3주 ≈ 20h 인력 / CC 집필 보조 별도.

---

## 완료/진행 중 (참조용)

- Phase 0-1 완료 (IREE 빌드 + SCALE-Sim TinyConv 손계산 검증)
- Phase 2 walking skeleton ex02 — 125 타일 sweep 완료 (prefetch:compute ≈ 2.5:1 측정)
- `/plan-eng-review` 리뷰 완료 (2026-04-21): 15 이슈 + 4 cross-model tension 처리됨

---

## 보관 (거부된 제안)

- ~IREE `--transform-file-name` upstream 수정/모니터링~ — 학부 범위 이탈, skip 결정 (2026-04-21).
- ~Timeloop/CoSA mapper baseline 추가~ — 학회 기여도 vs 시간 trade-off에서 현 포지셔닝 유지 결정 (2026-04-21). 논문 future work로 한 단락만 처리.
