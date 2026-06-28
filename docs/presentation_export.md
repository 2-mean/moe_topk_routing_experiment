# 발표자료 내보내기 가이드

## HTML (히트맵 Atlas, 권장)

[`sparse32_presentation.html`](sparse32_presentation.html) — **Plotly 인터랙티브 8×8 히트맵 6종 × 2 budget** + 복합 차트 3장

포함 히트맵:
- Mismatch Δloss · Asymmetry · Top-1 Agreement
- Nestedness · Spearman · Nestedness Excess

각 히트맵: 셀 수치 표시 · hover 상세값 · 읽는 법 · key pair 표

```bash
# 데이터/ HTML 재생성
python scripts/export_heatmap_data.py
python scripts/build_presentation_html.py

# 브라우저에서 열기
start docs/sparse32_presentation.html
# 또는
python -m http.server 8080
# → http://localhost:8080/docs/sparse32_presentation.html
```

PNG 히트맵 추가 생성 (matplotlib 필요, 서버에서 실행):
```bash
python scripts/generate_report_heatmaps.py   # 07–09 nestedness/spearman/excess 포함
```

---

## Marp (PDF/PPTX)

슬라이드 원본: [`sparse32_presentation.md`](sparse32_presentation.md) (Marp 형식)

## 방법 1: VS Code / Cursor 확장 (권장)

1. [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) 설치
2. `docs/sparse32_presentation.md` 열기
3. 우측 상단 **Preview** 아이콘으로 미리보기
4. 명령 팔레트 → `Marp: Export Slide Deck` → PDF 또는 PPTX 선택

## 방법 2: CLI

```bash
npm install -g @marp-team/marp-cli
cd docs
marp sparse32_presentation.md --pdf -o sparse32_presentation.pdf
marp sparse32_presentation.md --pptx -o sparse32_presentation.pptx
```

## 슬라이드 구성 (약 18장, 15–20분 발표용)

| # | 내용 |
|---|---|
| 1 | Title |
| 2–3 | Motivation & Research Questions |
| 4–5 | Setup & Experimental Design |
| 6–8 | Metrics, 3-level alignment, 수치 |
| 9 | Figure: Top-1 agreement |
| 10–11 | Mismatch cost & heatmap |
| 12–13 | Asymmetry figures |
| 14–15 | Noise floor |
| 16 | Matched loss |
| 17–18 | Evidence & Limitations |
| 19 | Takeaway |
| 20 | Appendix |

## 발표 시 강조 포인트

1. **가장 강한 claim**: hi→lo mismatch (k=8→1: +1.507)
2. **방향 비대칭**: 8/8 seeds, 두 budget 재현
3. **세 수준 alignment**: cross-seed≈random은 sanity, same-seed across-k vs oracle gap이 핵심
4. **한계 명시**: scratch 모델, 인과 미검증

## 커스터마이즈

- 발표 시간이 짧으면: Routing Alignment 수치 슬라이드 + Noise floor 슬라이드 생략 가능 (Figure 슬라이드만으로 대체)
- 학번/이름: Title 슬라이드 하단에 추가
- 테마 변경: front matter의 `theme: default` → `gaia`, `uncover` 등
