"""Build HTML presentation using matplotlib PNGs from results/report_figures/."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/report_figures"
OUT = ROOT / "docs/sparse32_presentation.html"

# Same figures and captions as docs/sparse32_full_report.md
FIGURES = [
    ("fig-01", "01_mismatch_delta_comparison.png", "Figure 1 · Mismatch Δloss",
     "행=train_k, 열=infer_k. 빨강=양수(불일치로 손실 증가), 파랑=음수. 대각선=0(matched baseline). "
     "왼쪽 열 전체가 짙은 빨강 → hi→lo 비용이 크다."),
    ("fig-02", "02_asymmetry_direction.png", "Figure 2 · Directional Asymmetry",
     "δ(hi→lo) − δ(lo→hi). 빨강(상삼각) = hi→lo가 더 비싸다. k=8→k=1이 가장 짙다. 두 budget에서 동일 패턴."),
    ("fig-03", "03_asymmetry_gap_ci.png", "Figure 3 · Asymmetry by K-gap (95% CI)",
     "빨강 막대=hi→lo 평균, 파랑 막대=lo→hi 절댓값. 파란 선+CI=|asymmetry|. "
     "회색 점선=gap=1 선형 외삽. gap=7에서 CI lower bound가 선형 예측을 명확히 초과."),
    ("fig-04", "04_noise_floor_direction.png", "Figure 4 · Noise Floor (gap=1)",
     "빨강=hi→lo, 파랑=lo→hi. SNR 차트에서 점선(SNR=2) 위 빨강만 STRONG. lo→hi는 대부분 noise level."),
    ("fig-05", "05_top1_agreement_comparison.png", "Figure 5 · Top-1 Agreement",
     "색이 진할수록 top-1 expert 일치율 ↑. 하단 우측(high-k 간) 진함, k=1 행/열 옅음. random=0.031, oracle=1.0."),
    ("fig-06", "06_matched_loss_comparison.png", "Figure 6 · Matched Loss by k",
     "파란=fixed-step(k↑→loss↓), 빨강=same-compute(k=1 최저). k=4 교차점. "
     "k=8 matched(0.384) < k=1(0.447) 이지만 k=8→1 mismatch +1.507."),
    ("fig-07", "07_nestedness_comparison.png", "Figure 7 · Nestedness (Overlap Recall)",
     "smaller-k expert overlap recall within larger-k set. random baseline = max(k)/32. excess 0.19–0.59."),
    ("fig-08", "08_spearman_comparison.png", "Figure 8 · Spearman (Gate Logit Ranking)",
     "router logit ranking 상관. random=0, oracle=1.0. 인접 k (7-8): ~0.86–0.89. "
     "k=1 관련: ~0.41–0.56 — top1보다 ranking 보존 ↑."),
    ("fig-09", "09_nestedness_excess_comparison.png", "Figure 9 · Nestedness Excess",
     "obs − max(k)/32. cardinality 효과 제거 후 구조 신호. 4-8 pair excess ~0.59."),
]

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MoE Top-k Routing — 발표자료</title>
  <style>
    :root {
      --bg: #f6f8fa;
      --surface: #fff;
      --text: #1a1a2e;
      --muted: #57606a;
      --accent: #c0392b;
      --accent2: #2980b9;
      --border: #d0d7de;
      --sidebar-w: 220px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
    }
    nav {
      position: fixed; top: 0; left: 0; width: var(--sidebar-w); height: 100vh;
      background: var(--surface); border-right: 1px solid var(--border);
      padding: 1rem 0; overflow-y: auto; z-index: 100;
    }
    nav .logo {
      padding: 0 1rem 0.8rem; font-size: 0.82rem; font-weight: 700;
      color: var(--accent2); border-bottom: 1px solid var(--border); margin-bottom: 0.6rem;
    }
    nav a {
      display: block; padding: 0.35rem 1rem; color: var(--muted);
      text-decoration: none; font-size: 0.78rem;
      border-left: 3px solid transparent;
    }
    nav a:hover, nav a.active { color: var(--text); background: #eef2f7; border-left-color: var(--accent2); }
    nav .nav-group {
      padding: 0.5rem 1rem 0.2rem; font-size: 0.65rem;
      text-transform: uppercase; letter-spacing: 0.05em; color: #888;
    }
    main { margin-left: var(--sidebar-w); padding: 2rem 2.5rem 4rem; max-width: 1100px; }
    .hero {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 2rem 2.5rem; margin-bottom: 2rem;
    }
    .hero h1 { font-size: 1.75rem; margin-bottom: 0.4rem; }
    .hero .sub { color: var(--muted); font-size: 1rem; }
    .hero .meta { margin-top: 1rem; font-size: 0.85rem; color: #888; }
    section { margin-bottom: 2.5rem; }
    h2 {
      font-size: 1.25rem; margin-bottom: 0.5rem; color: var(--text);
    }
    .figure-block {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 1.25rem 1.5rem 1.5rem;
      margin-bottom: 1.5rem;
    }
    .figure-block img {
      width: 100%; height: auto; display: block;
      border-radius: 4px; cursor: zoom-in;
      background: #fff;
    }
    .figure-block .caption {
      margin-top: 0.75rem; font-size: 0.88rem; color: var(--muted);
      padding: 0.65rem 0.85rem; background: #f0f4f8;
      border-left: 3px solid var(--accent2); border-radius: 0 4px 4px 0;
    }
    .figure-block .caption strong { color: var(--text); }
    .card-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem; margin: 1rem 0;
    }
    .stat {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.85rem 1rem;
    }
    .stat .val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
    .stat .lbl { font-size: 0.75rem; color: var(--muted); }
    .takeaway {
      background: #fff8f7; border: 1px solid #f5c6c0;
      border-radius: 8px; padding: 1.5rem; font-size: 1rem; line-height: 1.7;
    }
    .takeaway em { color: var(--accent); font-style: normal; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
    th { background: #eef2f7; text-align: left; padding: 0.45rem 0.65rem; }
    td { padding: 0.4rem 0.65rem; border-bottom: 1px solid var(--border); }
    #lightbox {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9);
      z-index: 999; align-items: center; justify-content: center; cursor: zoom-out;
    }
    #lightbox.open { display: flex; }
    #lightbox img { max-width: 98vw; max-height: 98vh; background: #fff; border-radius: 4px; }
    @media (max-width: 900px) { nav { display: none; } main { margin-left: 0; } }
  </style>
</head>
<body>
  <nav>
    <div class="logo">MoE Top-k<br>Routing</div>
    <a href="#intro">개요</a>
    <div class="nav-group">Figures (matplotlib)</div>
    __NAV_FIGURES__
    <div class="nav-group">결론</div>
    <a href="#evidence">Evidence</a>
    <a href="#takeaway">Takeaway</a>
  </nav>

  <main>
    <div class="hero" id="intro">
      <h1>MoE Top-k Routing 실험</h1>
      <p class="sub">Train-time k vs Inference-time k — Sparse32 TinyMoE, 8 seeds, two-budget</p>
      <p class="meta">그림 출처: <code>results/report_figures/</code> (matplotlib · <code>generate_report_heatmaps.py</code>)</p>
      <div class="card-grid">
        <div class="stat"><div class="val">+1.507</div><div class="lbl">k=8→1 mismatch</div></div>
        <div class="stat"><div class="val">8/8</div><div class="lbl">hi→lo &gt; lo→hi</div></div>
        <div class="stat"><div class="val">5–22×</div><div class="lbl">top1 above random</div></div>
      </div>
    </div>

    __FIGURE_SECTIONS__

    <section id="evidence">
      <h2>Evidence Summary</h2>
      <div class="figure-block">
        <table>
          <tr><th>Claim</th><th>강도</th><th>핵심 수치</th></tr>
          <tr><td>hi→lo &gt; lo→hi</td><td><strong>가장 강함</strong></td><td>8/8 unanimity</td></tr>
          <tr><td>k=8→1 mismatch</td><td><strong>강함</strong></td><td>+1.507 / +1.613</td></tr>
          <tr><td>routing priority 재구성</td><td><strong>강함</strong></td><td>oracle gap 0.30–0.85</td></tr>
          <tr><td>Spearman 7-8</td><td><strong>강함</strong></td><td>0.86–0.89 ranking 보존</td></tr>
        </table>
      </div>
    </section>

    <section id="takeaway">
      <h2>Takeaway</h2>
      <div class="takeaway">
        <p><em>High-k로 학습한 MoE를 low-k로 추론(hi→lo)하면 validation loss가 크게 악화된다.</em></p>
        <p style="margin-top:0.6rem">k=8 matched(0.384) &lt; k=1(0.447) 이지만 k=8→1 mismatch <em>+1.507</em>. 두 budget, 8 seeds 재현.</p>
      </div>
    </section>
  </main>

  <div id="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src="" alt=""></div>
  <script>
    function openLightbox(src) {
      document.getElementById('lightbox-img').src = src;
      document.getElementById('lightbox').classList.add('open');
    }
    function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
    document.querySelectorAll('.figure-block img').forEach(img => {
      img.addEventListener('click', () => openLightbox(img.src));
    });
    const navLinks = document.querySelectorAll('nav a[href^="#fig"]');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    document.querySelectorAll('.figure-block[id]').forEach(el => observer.observe(el));
  </script>
</body>
</html>"""

nav_figs = "\n".join(
    f'    <a href="#{fid}">{title.split("·")[0].strip()}</a>'
    for fid, _, title, _ in FIGURES
)

figure_sections = "\n".join(
    f"""    <section class="figure-block" id="{fid}">
      <h2>{title}</h2>
      <img src="../results/report_figures/{fname}" alt="{title}" loading="lazy">
      <p class="caption"><strong>읽는 법:</strong> {caption}</p>
    </section>"""
    for fid, fname, title, caption in FIGURES
)

html = HTML.replace("__NAV_FIGURES__", nav_figs).replace("__FIGURE_SECTIONS__", figure_sections)
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT}")

missing = [f for _, f, _, _ in FIGURES if not (FIG / f).exists()]
if missing:
    print("warning: missing PNGs (run python scripts/generate_report_heatmaps.py):", missing)
