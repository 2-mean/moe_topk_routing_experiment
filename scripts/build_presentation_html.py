"""Build self-contained HTML presentation with embedded heatmap data."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/report_figures/heatmap_data.json"
OUT = ROOT / "docs/sparse32_presentation.html"

data_json = DATA.read_text(encoding="utf-8")

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MoE Top-k Routing — 발표자료</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0f1117;
      --surface: #1a1d27;
      --surface2: #242836;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --accent: #e74c3c;
      --accent2: #3498db;
      --green: #2ecc71;
      --border: #2d3348;
      --sidebar-w: 240px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }
    nav {
      position: fixed; top: 0; left: 0; width: var(--sidebar-w); height: 100vh;
      background: var(--surface); border-right: 1px solid var(--border);
      padding: 1.2rem 0; overflow-y: auto; z-index: 100;
    }
    nav .logo { padding: 0 1.2rem 1rem; font-size: 0.85rem; font-weight: 700; color: var(--accent2); border-bottom: 1px solid var(--border); margin-bottom: 0.8rem; }
    nav a {
      display: block; padding: 0.45rem 1.2rem; color: var(--muted); text-decoration: none; font-size: 0.82rem;
      border-left: 3px solid transparent;
    }
    nav a:hover, nav a.active { color: var(--text); background: var(--surface2); border-left-color: var(--accent2); }
    nav .nav-group { padding: 0.6rem 1.2rem 0.25rem; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #666; }
    main { margin-left: var(--sidebar-w); padding: 2rem 2.5rem 4rem; max-width: 1400px; }
    .hero {
      background: linear-gradient(135deg, #1a1d27 0%, #2c3e50 100%);
      border-radius: 12px; padding: 2.5rem 3rem; margin-bottom: 2.5rem;
      border: 1px solid var(--border);
    }
    .hero h1 { font-size: 2rem; margin-bottom: 0.5rem; }
    .hero .sub { color: var(--muted); font-size: 1.05rem; max-width: 720px; }
    .hero .meta { margin-top: 1.2rem; font-size: 0.85rem; color: #777; }
    section { margin-bottom: 3rem; }
    h2 {
      font-size: 1.45rem; margin-bottom: 0.6rem; padding-bottom: 0.4rem;
      border-bottom: 2px solid var(--accent2); display: inline-block;
    }
    h3 { font-size: 1.05rem; margin: 1.2rem 0 0.5rem; color: var(--accent2); }
    p, li { color: #ccc; font-size: 0.95rem; }
    ul { padding-left: 1.3rem; margin: 0.5rem 0; }
    li { margin-bottom: 0.35rem; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 1.4rem 1.6rem; margin-top: 1rem;
    }
    .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .stat { background: var(--surface2); border-radius: 8px; padding: 1rem 1.2rem; }
    .stat .val { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
    .stat .lbl { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.8rem; }
    th { background: var(--surface2); text-align: left; padding: 0.5rem 0.7rem; color: var(--muted); font-weight: 600; }
    td { padding: 0.45rem 0.7rem; border-bottom: 1px solid var(--border); }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .heatmap-block { margin-top: 2rem; }
    .heatmap-block .desc { color: var(--muted); font-size: 0.88rem; margin: 0.4rem 0 0.8rem; max-width: 900px; }
    .heatmap-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    @media (max-width: 1100px) { .heatmap-row { grid-template-columns: 1fr; } nav { display: none; } main { margin-left: 0; } }
    .plot-box {
      background: #fff; border-radius: 8px; overflow: hidden;
      min-height: 480px; border: 1px solid var(--border);
    }
    .plot-label { font-size: 0.8rem; color: var(--muted); text-align: center; margin-bottom: 0.4rem; font-weight: 600; }
    .guide {
      background: #1e2433; border-left: 4px solid var(--accent2);
      padding: 0.9rem 1.1rem; margin-top: 0.8rem; border-radius: 0 8px 8px 0; font-size: 0.88rem;
    }
    .guide strong { color: var(--accent2); }
    .guide ul { margin-top: 0.4rem; }
    .figure-full img {
      width: 100%; border-radius: 8px; border: 1px solid var(--border);
      cursor: zoom-in; background: #fff;
    }
    .figure-full .caption { font-size: 0.82rem; color: var(--muted); margin-top: 0.5rem; }
    .levels {
      text-align: center; font-size: 1rem; padding: 1rem;
      background: var(--surface2); border-radius: 8px; margin: 1rem 0;
    }
    .levels span { display: inline-block; margin: 0 0.3rem; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .l-rand { background: #333; color: #aaa; }
    .l-cross { background: #444; color: #bbb; }
    .l-same { background: #1a5276; color: #aed6f1; }
    .l-oracle { background: #1e8449; color: #abebc6; }
    .takeaway {
      background: linear-gradient(135deg, #1a2744, #2c1a1a);
      border: 1px solid var(--accent); border-radius: 12px;
      padding: 2rem; font-size: 1.05rem; line-height: 1.7;
    }
    .takeaway em { color: var(--accent); font-style: normal; font-weight: 700; }
    #lightbox {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92);
      z-index: 999; align-items: center; justify-content: center; cursor: zoom-out;
    }
    #lightbox.open { display: flex; }
    #lightbox img { max-width: 96vw; max-height: 96vh; border-radius: 6px; }
    .tag { display: inline-block; background: var(--accent2); color: #fff; font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.3rem; vertical-align: middle; }
  </style>
</head>
<body>
  <nav>
    <div class="logo">MoE Top-k<br>Routing 실험</div>
    <a href="#intro">개요</a>
    <a href="#design">실험 설계</a>
    <div class="nav-group">히트맵 Atlas</div>
    <a href="#hm-mismatch">Mismatch Δloss</a>
    <a href="#hm-asymmetry">Asymmetry</a>
    <a href="#hm-top1">Top-1 Agreement</a>
    <a href="#hm-nestedness">Nestedness</a>
    <a href="#hm-spearman">Spearman</a>
    <a href="#hm-excess">Nestedness Excess</a>
    <div class="nav-group">복합 차트</div>
    <a href="#fig-asym-ci">Asymmetry CI</a>
    <a href="#fig-noise">Noise Floor</a>
    <a href="#fig-loss">Matched Loss</a>
    <div class="nav-group">결론</div>
    <a href="#evidence">Evidence</a>
    <a href="#takeaway">Takeaway</a>
  </nav>

  <main>
    <div class="hero" id="intro">
      <h1>MoE Top-k Routing 실험</h1>
      <p class="sub">Train-time k와 Inference-time k 불일치가 routing structure와 validation loss에 미치는 영향 — Sparse32 TinyMoE, 8 seeds, two-budget 설계</p>
      <p class="meta">Fixed-step (1500 steps) · Same-compute (k×step) · 512 runs · RTX 3080 Ti</p>
    </div>

    <section id="design">
      <h2>Research Questions &amp; Setup</h2>
      <div class="card-grid">
        <div class="card"><strong>(a)</strong> k<sub>train</sub>≠k<sub>infer</sub>일 때 routing은 얼마나 diverge?<br><strong>(b)</strong> loss 영향의 크기와 <em>방향</em>?<br><strong>(c)</strong> training stochasticity를 초과하는 k 효과?</div>
        <div class="card">TinyMoE 8L · 32 experts · d=192<br>k∈{1..8} grid · 8 seeds<br>합성 7-category 데이터</div>
      </div>
      <div class="levels">
        <span class="l-rand">random 0.031</span> ≈
        <span class="l-cross">cross-seed 0.031</span> ≪
        <span class="l-same">same-seed across-k 0.15–0.70</span> ≪
        <span class="l-oracle">oracle 1.0</span>
      </div>
      <div class="card-grid" style="margin-top:1rem">
        <div class="stat"><div class="val">+1.507</div><div class="lbl">k=8→1 mismatch (fixed-step)</div></div>
        <div class="stat"><div class="val">8/8</div><div class="lbl">seeds hi→lo &gt; lo→hi</div></div>
        <div class="stat"><div class="val">5–22×</div><div class="lbl">top1 above random</div></div>
        <div class="stat"><div class="val">0.30–0.85</div><div class="lbl">oracle gap (priority 재구성)</div></div>
      </div>
    </section>

    <!-- HEATMAP SECTIONS rendered by JS -->
    <section id="heatmap-atlas"><h2>Interactive Heatmap Atlas</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:1.5rem">각 셀에 수치 표시 · hover로 정확한 값 확인 · Fixed-step / Same-compute 나란히 비교</p>
      <div id="heatmap-sections"></div>
    </section>

    <section id="fig-asym-ci" class="figure-full">
      <h2>Asymmetry by K-gap — 95% CI vs Linear</h2>
      <img src="../results/report_figures/03_asymmetry_gap_ci.png" alt="Asymmetry CI" onclick="openLightbox(this.src)">
      <p class="caption">빨강=hi→lo, 파랑=lo→hi | 파란선=|asymmetry|+CI | 회색점선=gap=1 선형 외삽 · gap=7에서 CI lower &gt; linear</p>
    </section>

    <section id="fig-noise" class="figure-full">
      <h2>Noise Floor — Adjacent-k (gap=1)</h2>
      <img src="../results/report_figures/04_noise_floor_direction.png" alt="Noise floor" onclick="openLightbox(this.src)">
      <p class="caption">hi→lo SNR 3–15 (STRONG) · lo→hi SNR &lt;2 (noise) · "인접 k 안전"은 lo→hi에만 해당</p>
    </section>

    <section id="fig-loss" class="figure-full">
      <h2>Matched Inference Loss by k</h2>
      <img src="../results/report_figures/06_matched_loss_comparison.png" alt="Matched loss" onclick="openLightbox(this.src)">
      <p class="caption">k=4 교차점 · k=8 matched(0.384) &lt; k=1(0.447) 이지만 k=8→1 mismatch +1.507</p>
    </section>

    <section id="evidence">
      <h2>Evidence Summary</h2>
      <div class="card">
        <table>
          <tr><th>Claim</th><th>강도</th><th>핵심 수치</th></tr>
          <tr><td>k가 routing priority 재구성</td><td class="num" style="color:var(--green)">강함</td><td>oracle gap 0.30–0.85</td></tr>
          <tr><td>hi→lo &gt; lo→hi</td><td class="num" style="color:var(--accent)">가장 강함</td><td>8/8 unanimity, 5–13×</td></tr>
          <tr><td>k=8→1 mismatch</td><td class="num" style="color:var(--green)">강함</td><td>+1.507 / +1.613</td></tr>
          <tr><td>nestedness excess</td><td class="num" style="color:var(--green)">중간-강함</td><td>+0.19 – +0.59 above b/E</td></tr>
          <tr><td>lo→hi gap=1 safe</td><td>중간</td><td>SNR &lt; 2</td></tr>
          <tr><td>two-budget 재현</td><td class="num" style="color:var(--green)">강함</td><td>방향 동일</td></tr>
        </table>
      </div>
    </section>

    <section id="takeaway">
      <h2>Takeaway</h2>
      <div class="takeaway">
        <p><em>High-k로 학습한 MoE를 low-k로 추론(hi→lo)하면 validation loss가 크게 악화된다.</em></p>
        <p style="margin-top:0.8rem">k=8 matched loss(0.384)가 k=1(0.447)보다 낮아도, k=8→1 mismatch는 <em>+1.507</em>. 이 비대칭성은 fixed-step · same-compute 두 budget, 8 seeds 모두에서 재현.</p>
      </div>
    </section>
  </main>

  <div id="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src="" alt=""></div>

  <script id="heatmap-data" type="application/json">""" + data_json + r"""</script>
  <script>
    const DATA = JSON.parse(document.getElementById('heatmap-data').textContent);
    const LABELS = DATA.labels;

    const HEATMAP_SPECS = [
      {
        id: 'hm-mismatch', title: 'Mismatch Δloss (8×8)',
        key: 'mismatch_delta', colorscale: 'RdBu', reversescale: true, zmid: 0,
        fmt: '.3f', symmetric: true,
        desc: '행=train_k, 열=infer_k. 대각선=0(matched). 빨강=불일치로 loss 증가.',
        guide: ['<strong>왼쪽 열(hi→lo)</strong> 전체가 짙은 빨강 — k=8→1: +1.507', '오른쪽 열(lo→hi)은 상대적으로 약함', '두 budget에서 동일한 비대칭 패턴'],
        keyCells: [['8','1','+1.507','+1.613'],['4','1','+0.890','+1.123'],['1','8','+0.300','+0.124']]
      },
      {
        id: 'hm-asymmetry', title: 'Directional Asymmetry δ(hi→lo)−δ(lo→hi)',
        key: 'asymmetry', colorscale: 'RdBu', reversescale: true, zmid: 0,
        fmt: '.2f', symmetric: true,
        desc: '양수(빨강)=hi→lo가 더 비싸다. k=8↔1 쌍이 최대.',
        guide: ['8/8 seeds unanimity (gap≥2)', 'gap=7 asymmetry: fixed 1.21, same-compute 1.49', '95% CI가 gap=7에서 linear extrapolation 초과'],
        keyCells: [['8','1','+1.51','+1.61'],['7','1','+1.38','+1.52'],['2','1','+0.22','+0.19']]
      },
      {
        id: 'hm-top1', title: 'Top-1 Expert Agreement',
        key: 'top1_agreement', colorscale: 'Blues', zmin: 0, zmax: 1,
        fmt: '.3f', symmetric: false,
        desc: 'same-seed, different-k 모델 간 top-1 expert 일치율. random=0.031, oracle=1.0.',
        guide: ['하단 우측(high-k 간) 진함 — 7-8 pair ~0.65', 'k=1 관련 행/열 옅음 — ~0.23 (random의 7×)', '인접 k일수록 agreement ↑'],
        keyCells: [['7','8','0.645','0.695'],['1','8','0.231','0.163'],['4','8','0.526','0.533']]
      },
      {
        id: 'hm-nestedness', title: 'Nestedness (Overlap Recall)',
        key: 'nestedness', colorscale: 'YlGnBu', zmin: 0, zmax: 1,
        fmt: '.3f', symmetric: false,
        desc: 'smaller-k expert가 larger-k set에 포함되는 per-token recall. random baseline = max(k)/32.',
        guide: ['k=1-8 pair: 0.73 (random 0.25 대비 +0.48)', 'nestedness excess 0.19–0.59 — cardinality 이상의 구조', 'routing alignment와 mismatch cost 음의 상관 (−0.79)'],
        keyCells: [['1','8','0.732','0.635'],['4','8','0.835','0.844'],['7','8','0.804','0.838']]
      },
      {
        id: 'hm-spearman', title: 'Spearman (Gate Logit Ranking)',
        key: 'spearman', colorscale: 'PuRd', zmin: 0, zmax: 1,
        fmt: '.3f', symmetric: false,
        desc: 'router gate logit ranking 상관. random=0, oracle cutoff=1.0.',
        guide: ['인접 k pair (7-8): ~0.86 — ranking 거의 보존', 'k=1 관련: ~0.56 — logit ordering 분기', 'top1보다 ranking correlation이 더 높음'],
        keyCells: [['7','8','0.860','0.875'],['1','2','0.560','0.485'],['4','5','0.817','0.812']]
      },
      {
        id: 'hm-excess', title: 'Nestedness Excess (obs − max k/32)',
        key: 'nestedness_excess', colorscale: 'Oranges', zmin: 0,
        fmt: '.3f', symmetric: false,
        desc: 'uniform random baseline 대비 초과 nestedness. cardinality 효과 제거 후 구조 신호.',
        guide: ['1-8 excess: +0.48 (fixed), +0.39 (same-compute)', '4-8 excess: +0.59 — 가장 큰 구조적 divergence', 'Claim B 핵심: 차이가 cardinality만이 아님'],
        keyCells: [['4','8','+0.585','+0.594'],['1','4','+0.407','+0.294'],['1','2','+0.295','+0.188']]
      },
    ];

    function matrixText(z, fmt) {
      const dec = (fmt.match(/\.(\d)/) || [, '3'])[1];
      return z.map(row => row.map(v => v == null ? '' : Number(v).toFixed(Number(dec))));
    }

    function plotHeatmap(divId, z, spec, budgetLabel) {
      const layout = {
        title: { text: budgetLabel, font: { size: 13 } },
        xaxis: { title: 'train_k (B)', tickvals: [0,1,2,3,4,5,6,7], ticktext: LABELS, side: 'bottom' },
        yaxis: { title: 'train_k (A)', tickvals: [0,1,2,3,4,5,6,7], ticktext: LABELS, autorange: 'reversed' },
        margin: { l: 70, r: 30, t: 40, b: 70 },
        paper_bgcolor: '#fff', plot_bgcolor: '#fff',
        height: 460,
      };
      const trace = {
        z, x: LABELS, y: LABELS, type: 'heatmap',
        colorscale: spec.colorscale,
        reversescale: spec.reversescale || false,
        zmid: spec.zmid,
        zmin: spec.zmin,
        zmax: spec.zmax,
        text: matrixText(z, spec.fmt),
        texttemplate: '%{text}',
        textfont: { size: 10, color: '#222' },
        hovertemplate: 'k_a=%{y}<br>k_b=%{x}<br>value=%{z:.4f}<extra></extra>',
        showscale: true,
      };
      Plotly.newPlot(divId, [trace], layout, { responsive: true, displayModeBar: false });
    }

    function renderSections() {
      const container = document.getElementById('heatmap-sections');
      HEATMAP_SPECS.forEach(spec => {
        const block = document.createElement('div');
        block.className = 'heatmap-block';
        block.id = spec.id;
        let keyRows = spec.keyCells.map(([a,b,fs,mc]) =>
          `<tr><td>k=${a} vs k=${b}</td><td class="num">${fs}</td><td class="num">${mc}</td></tr>`
        ).join('');
        block.innerHTML = `
          <h3>${spec.title}</h3>
          <p class="desc">${spec.desc}</p>
          <div class="heatmap-row">
            <div><div class="plot-label">Fixed-step</div><div class="plot-box" id="plot-${spec.id}-fs"></div></div>
            <div><div class="plot-label">Same-compute</div><div class="plot-box" id="plot-${spec.id}-mc"></div></div>
          </div>
          <div class="guide"><strong>읽는 법</strong><ul>${spec.guide.map(g=>'<li>'+g+'</li>').join('')}</ul></div>
          <table><tr><th>Key pair</th><th class="num">Fixed-step</th><th class="num">Same-compute</th></tr>${keyRows}</table>
        `;
        container.appendChild(block);
        plotHeatmap(`plot-${spec.id}-fs`, DATA.fixed_step[spec.key], spec, 'Fixed-step 8-seed');
        plotHeatmap(`plot-${spec.id}-mc`, DATA.same_compute[spec.key], spec, 'Same-compute 8-seed');
      });
    }

    function openLightbox(src) {
      document.getElementById('lightbox-img').src = src;
      document.getElementById('lightbox').classList.add('open');
    }
    function closeLightbox() {
      document.getElementById('lightbox').classList.remove('open');
    }
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

    // sidebar active link
    const navLinks = document.querySelectorAll('nav a');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
        }
      });
    }, { rootMargin: '-30% 0px -60% 0px' });
    document.querySelectorAll('section[id], .heatmap-block[id]').forEach(el => observer.observe(el));

    renderSections();
  </script>
</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}")
