# 발표자료 / 시각자료

## PNG (matplotlib — MD 보고서와 동일)

모든 그림: **`results/report_figures/`**

```bash
python scripts/build_visual_assets.py
# 또는
python scripts/generate_report_heatmaps.py
python scripts/build_presentation_html.py
```

| 파일 | 내용 |
|---|---|
| `01_mismatch_delta_comparison.png` | Mismatch Δloss 8×8 |
| `02_asymmetry_direction.png` | 방향 비대칭 |
| `03_asymmetry_gap_ci.png` | gap별 CI |
| `04_noise_floor_direction.png` | gap=1 noise floor |
| `05_top1_agreement_comparison.png` | Top-1 agreement |
| `06_matched_loss_comparison.png` | Matched loss |
| `07_nestedness_comparison.png` | Nestedness |
| `08_spearman_comparison.png` | Spearman |
| `09_nestedness_excess_comparison.png` | Nestedness excess |
| `presentation/*.png` | budget별 단독 히트맵 (발표용) |

## HTML 발표자료

[`sparse32_presentation.html`](sparse32_presentation.html) — **위 matplotlib PNG 그대로** 사용 (Plotly 없음)

```bash
start docs/sparse32_presentation.html
```

## Marp (선택)

[`sparse32_presentation.md`](sparse32_presentation.md) — 동일 PNG 경로 참조
