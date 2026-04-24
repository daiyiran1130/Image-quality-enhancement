#!/usr/bin/env python3
"""
Generate a pgfplots LaTeX figure: CI-based box plots grouped by PSNR scope.

5 models plotted:
  Diffusion — Ours, ResShift, SD-v1.5
  GAN       — MDA-Net  (best GAN by avg. PSNR)
  Trans.    — GFE-Net  (best Trans. by avg. PSNR)

4 metrics: PSNR, LPIPS, VIF, HARALICK
4 PSNR-scope groups: <20, 20-30, 30-40, >40

All pairwise comparisons vs. Ours are significant at p < 0.001 (***),
so no significance brackets are drawn; the fact is stated in the caption note.
"""

import re
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\work\figure_table\figure2\classify_with_PSNR\classify_with_PSNR.parquet"
OUT_PATHS = [
    r"D:\work\figure_table\figure2\classify_with_PSNR\Draw fig1.1.2.tex",
    r"C:\Users\冰糖雪梨\PycharmProjects\PythonProject\Draw fig1.1.2.tex",
]

TARGET_MODELS = ['Ours', 'ResShift', 'SD-v1.5', 'MDA-Net', 'GFE-Net']
SCOPE_ORDER   = ['PSNR<20', '(20, 30)', '(30, 40)', '(40, 40+)']
SCOPE_LABELS  = [r'$<$20', r'20--30', r'30--40', r'$>$40']

# (metric_col, y-axis label, subplot title, y_min, y_max)
METRICS = [
    ('PSNR',     'PSNR (dB)',  'PSNR',     25.5,  47.0),
    ('LPIPS',    'LPIPS',      'LPIPS',     0.01,  0.22),
    ('VIF',      'VIF',        'VIF',       0.44,  0.95),
    ('HARALICK', 'Haralick',   'Haralick',  0.10,  5.20),
]

# Visual style per model
MODEL_STYLE = {
    'Ours':     {'fill': 'myblue',          'draw': 'myblue!70!black',
                 'legend': r'Ours (Diffusion)'},
    'ResShift': {'fill': 'myblue!58!white', 'draw': 'myblue!80!black',
                 'legend': r'ResShift (Diffusion)'},
    'SD-v1.5':  {'fill': 'myblue!28!white', 'draw': 'myblue!70!black',
                 'legend': r'SD-v1.5 (Diffusion)'},
    'MDA-Net':  {'fill': 'myred',           'draw': 'myred!70!black',
                 'legend': r'MDA-Net (GAN)'},
    'GFE-Net':  {'fill': 'mygreen',         'draw': 'mygreen!70!black',
                 'legend': r'GFE-Net (Trans.)'},
}

# Box geometry: 5 models per group, 0.15 apart, width 0.12
OFFSETS    = [-0.30, -0.15, 0.00, 0.15, 0.30]
BOX_EXT    = 0.12


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_ci(s):
    nums = re.findall(r'[\d.]+', str(s))
    return float(nums[0]), float(nums[1])

def f3(x):
    return f'{x:.3f}'


# ── Load & index data ─────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df = df[df['model_name'].isin(TARGET_MODELS)]
lut = {(r['PSNR_scope'], r['model_name']): r for _, r in df.iterrows()}


# ── Build LaTeX ───────────────────────────────────────────────────────────────
L = []
def ln(*lines):
    L.extend(lines)

ln(
    r'% Auto-generated — edit draw_psnr_boxplots.py to regenerate',
    r'\documentclass[tikz,border=8pt]{standalone}',
    r'\usepackage{pgfplots}',
    r'\usepgfplotslibrary{statistics}',
    r'\usepgfplotslibrary{groupplots}',
    r'\usepgflibrary{patterns}',
    r'\usetikzlibrary{patterns,calc}',
    r'\pgfplotsset{compat=1.18}',
    r'',
    r'%% Colour palette (same as the main figure)',
    r'\definecolor{myblue}  {HTML}{2471A3}',
    r'\definecolor{myred}   {HTML}{C0392B}',
    r'\definecolor{mygreen} {HTML}{1E8449}',
    r'\definecolor{mygray}  {HTML}{7F8C8D}',
    r'\definecolor{myorange}{HTML}{D35400}',
    r'',
    r'\begin{document}',
    r'\begin{tikzpicture}',
    r'',
)

# ── groupplot environment ─────────────────────────────────────────────────────
xtick_str        = ','.join(str(i) for i in range(4))
xticklabels_str  = ','.join(SCOPE_LABELS)

ln(
    r'\begin{groupplot}[',
    r'    group style={',
    r'        group size=2 by 2,',
    r'        horizontal sep=1.9cm,',
    r'        vertical sep=2.1cm,',
    r'    },',
    r'    width=7.0cm, height=5.4cm,',
    r'    %% box geometry & direction',
    r'    boxplot/draw direction=y,',
    r'    boxplot={',
    f'        box extend={BOX_EXT},',
    r'        whisker extend=0,',
    r'        average=none,',
    r'    },',
    r'    %% x axis',
    f'    xtick={{{xtick_str}}},',
    f'    xticklabels={{{xticklabels_str}}},',
    r'    xmin=-0.55, xmax=3.55,',
    r'    %% style',
    r'    tick label style={font=\scriptsize},',
    r'    label style={font=\scriptsize},',
    r'    title style={font=\small\bfseries},',
    r'    xlabel style={font=\scriptsize},',
    r'    ylabel style={font=\scriptsize},',
    r'    ymajorgrids=true,',
    r'    grid style={line width=0.3pt, color=black!12},',
    r']',
    r'',
)

# ── Four metric subplots ──────────────────────────────────────────────────────
for m_idx, (metric, ylabel, title, ymin, ymax) in enumerate(METRICS):
    ci_col = f'{metric}_95CI'

    # \nextgroupplot options
    opts = [
        f'    title={{{title}}},',
        f'    ylabel={{{ylabel}}},',
        f'    ymin={ymin:.2f}, ymax={ymax:.2f},',
    ]
    # x-label only for bottom row
    if m_idx >= 2:
        opts.append(r'    xlabel={Input PSNR (dB)},')
    # legend to name: only first subplot
    if m_idx == 0:
        opts += [
            r'    legend to name=boxlegend,',
            r'    legend style={',
            r'        legend columns=5,',
            r'        column sep=6pt,',
            r'        /tikz/every odd column/.style={column sep=0pt},',
            r'        font=\scriptsize,',
            r'        draw=black!25,',
            r'        rounded corners=2pt,',
            r'        fill=white,',
            r'        inner sep=3pt,',
            r'        row sep=1pt,',
            r'    },',
        ]

    ln(r'\nextgroupplot[')
    ln(*opts)
    ln(r']')

    # Legend image entries (first subplot only)
    if m_idx == 0:
        for model in TARGET_MODELS:
            s = MODEL_STYLE[model]
            ln(
                r'\addlegendimage{fill=' + s['fill'] + r', draw=' + s['draw']
                + r', area legend, line width=0.7pt}',
                r'\addlegendentry{' + s['legend'] + r'}',
            )
        ln('')

    # ── Box plots: iterate scopes then models ──────────────────────────────
    for g, scope in enumerate(SCOPE_ORDER):
        for m, model in enumerate(TARGET_MODELS):
            row = lut.get((scope, model))
            if row is None:
                continue
            mean_val          = float(row[metric])
            ci_low, ci_high   = parse_ci(row[ci_col])
            draw_pos          = g + OFFSETS[m]
            s                 = MODEL_STYLE[model]

            ln(
                r'\addplot[',
                r'    boxplot prepared={',
                f'        lower whisker={f3(ci_low)},',
                f'        lower quartile={f3(ci_low)},',
                f'        median={f3(mean_val)},',
                f'        upper quartile={f3(ci_high)},',
                f'        upper whisker={f3(ci_high)},',
                f'        draw position={f3(draw_pos)},',
                f'        box extend={BOX_EXT},',
                r'    },',
                f'    fill={s["fill"]}, draw={s["draw"]}, line width=0.7pt,',
                r'] coordinates {};',
            )
        ln('')  # blank between groups

    ln('')  # blank between subplots

ln(r'\end{groupplot}', r'')

# ── Legend node below figure ──────────────────────────────────────────────────
ln(
    r'%% Legend placed below the figure',
    r'\node[anchor=north, inner sep=0pt] at',
    r'    ($(current bounding box.south)+(0,-0.45cm)$)',
    r'    {\pgfplotslegendfromname{boxlegend}};',
    r'',
    r'%% Caption note: significance',
    r'\node[anchor=north, font=\scriptsize, text=black!65, text width=14cm,',
    r'      align=center] at',
    r'    ($(current bounding box.south)+(0,-1.20cm)$)',
    r'    {All pairwise comparisons vs.\ \textbf{Ours} are significant at',
    r'     $p < 0.001$ ($^{***}$, Wilcoxon signed-rank test).};',
    r'',
    r'\end{tikzpicture}',
    r'\end{document}',
)

# ── Write output ──────────────────────────────────────────────────────────────
output = '\n'.join(L)
for path in OUT_PATHS:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(output)
    print(f'Written: {path}')
print(f'Lines:   {len(L)}')
