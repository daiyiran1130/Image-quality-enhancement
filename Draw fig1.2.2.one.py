#!/usr/bin/env python3
"""
Draw fig1.2.2.one.py — CI-box figure for SINGLE degradation types.

Single degradation types (5): blur | color | halo | hole | spot
5 models : Ours, ResShift, SD-v1.5 (Diffusion) | MDA-Net (GAN) | GFE-Net (Trans.)
4 metrics: PSNR, LPIPS, VIF, Haralick
Layout   : 2x2 groupplot, x-axis = 5 single degradation types
Draws CI boxes + mean lines using pure TikZ filldraw (no statistics library).
"""

import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\work\figure_table\figure2\classify_with_degradation\classify_with_degradation.parquet"
OUT_PATHS = [
    r"D:\work\figure_table\figure2\classify_with_degradation\Draw fig1.2.2.one.tex",
    r"C:\Users\冰糖雪梨\PycharmProjects\PythonProject\Draw fig1.2.2.one.tex",
]

# ── Data config ───────────────────────────────────────────────────────────────
TARGET_MODELS = ['Ours', 'ResShift', 'SD-v1.5', 'MDA-Net', 'GFE-Net']
SCOPE_ORDER   = ['blur', 'color', 'halo', 'hole', 'spot']
SCOPE_LABELS  = ['blur', 'color', 'halo', 'hole', 'spot']

# (col, y-label, title, ymin, ymax)
METRICS = [
    ('PSNR',     'PSNR (dB)', 'PSNR',    25.5, 47.0),
    ('LPIPS',    'LPIPS',     'LPIPS',    0.01, 0.22),
    ('VIF',      'VIF',       'VIF',      0.44, 0.95),
    ('HARALICK', 'Haralick',  'Haralick', 0.10, 5.20),
]

MODEL_STYLE = {
    'Ours':     {'fill': 'myblue',          'draw': 'myblue!70!black',
                 'legend': 'Ours (Diffusion)'},
    'ResShift': {'fill': 'myblue!58!white', 'draw': 'myblue!80!black',
                 'legend': 'ResShift (Diffusion)'},
    'SD-v1.5':  {'fill': 'myblue!28!white', 'draw': 'myblue!70!black',
                 'legend': 'SD-v1.5 (Diffusion)'},
    'MDA-Net':  {'fill': 'myred',           'draw': 'myred!70!black',
                 'legend': 'MDA-Net (GAN)'},
    'GFE-Net':  {'fill': 'mygreen',         'draw': 'mygreen!70!black',
                 'legend': 'GFE-Net (Trans.)'},
}

# Box geometry: 5 models x 0.20 spacing
OFFSETS = [-0.40, -0.20, 0.00, 0.20, 0.40]
HALF    = 0.098      # half box width
BOX_LW  = '1.0pt'
MEAN_LW = '1.8pt'


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_ci(s):
    nums = re.findall(r'[\d.]+', str(s))
    return float(nums[0]), float(nums[1])

def f3(x):
    return '{:.3f}'.format(x)


# ── Load data ─────────────────────────────────────────────────────────────────
df  = pd.read_parquet(DATA_PATH)
df  = df[df['model_name'].isin(TARGET_MODELS)]
lut = {(r['degradations_applied'], r['model_name']): r for _, r in df.iterrows()}


# ── Build LaTeX ───────────────────────────────────────────────────────────────
L = []
def ln(*lines):
    L.extend(lines)

n_scopes        = len(SCOPE_ORDER)
xtick_str       = ','.join(str(i) for i in range(n_scopes))
xticklabels_str = ','.join(SCOPE_LABELS)

# ── Preamble ──────────────────────────────────────────────────────────────────
ln(
    '% Auto-generated -- edit "Draw fig1.2.2.one.py" to regenerate',
    r'\documentclass[tikz,border=8pt]{standalone}',
    r'\usepackage{fontspec}',
    r'\setmainfont{Arial}',
    r'\setsansfont{Arial}',
    r'\usepackage{pgfplots}',
    r'\usepgfplotslibrary{groupplots}',
    r'\usetikzlibrary{calc}',
    r'\pgfplotsset{compat=1.16}',
    r'',
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

# ── groupplot ─────────────────────────────────────────────────────────────────
ln(
    r'\begin{groupplot}[',
    r'    group style={',
    r'        group size=2 by 2,',
    r'        horizontal sep=2.2cm,',
    r'        vertical sep=2.3cm,',
    r'    },',
    r'    width=15.0cm, height=7.0cm,',
    '    xtick={' + xtick_str + '},',
    '    xticklabels={' + xticklabels_str + '},',
    '    xmin=-0.55, xmax=' + f3(n_scopes - 1 + 0.55) + ',',
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

# ── Four subplots ─────────────────────────────────────────────────────────────
for m_idx, (metric, ylabel, title, ymin, ymax) in enumerate(METRICS):
    ci_col = metric + '_95CI'

    ln(r'\nextgroupplot[')
    ln('    title={' + title + '},')
    ln('    ylabel={' + ylabel + '},')
    ln('    ymin=' + f3(ymin) + ', ymax=' + f3(ymax) + ',')
    if m_idx >= 2:
        ln(r'    xlabel={Degradation Type},')
    ln(r']')

    ln(r'\addplot[draw=none] coordinates {(-0.55,' + f3(ymin) + ')};')
    ln('')

    for g, scope in enumerate(SCOPE_ORDER):
        for m, model in enumerate(TARGET_MODELS):
            row = lut.get((scope, model))
            if row is None:
                continue
            mean_val        = float(row[metric])
            ci_low, ci_high = parse_ci(row[ci_col])
            cx              = g + OFFSETS[m]
            xl, xr          = cx - HALF, cx + HALF
            s               = MODEL_STYLE[model]

            ln(
                r'\filldraw[fill=' + s['fill'] + ', draw=' + s['draw']
                + ', line width=' + BOX_LW + ']',
                '    (axis cs:' + f3(xl) + ',' + f3(ci_low) + ')'
                + ' rectangle '
                + '(axis cs:' + f3(xr) + ',' + f3(ci_high) + ');',
            )
            ln(
                r'\draw[' + s['draw'] + ', line width=' + MEAN_LW + ']',
                '    (axis cs:' + f3(xl) + ',' + f3(mean_val)
                + ') -- (axis cs:' + f3(xr) + ',' + f3(mean_val) + ');',
            )
        ln('')
    ln('')

ln(r'\end{groupplot}', r'')

# ── Legend ────────────────────────────────────────────────────────────────────
def swatch(s):
    return (r'\tikz[baseline=-1pt]\filldraw[fill=' + s['fill']
            + r',draw=' + s['draw']
            + r',line width=0.5pt](0,0)rectangle(0.32cm,0.17cm);')

entries = [(m, MODEL_STYLE[m]) for m in TARGET_MODELS]

ln(
    r'\node[draw=black!25, rounded corners=2pt, fill=white, inner sep=5pt,',
    r'      anchor=north, font=\scriptsize] at',
    r'    ($(current bounding box.south)+(0,-0.5cm)$) {%',
    r'  \begin{tabular}{@{}lll@{}}',
)
r1 = [swatch(s) + r'~' + s['legend'] for _, s in entries[:3]]
ln('    ' + r' & '.join(r1) + r' \\[2pt]')
r2 = [swatch(s) + r'~' + s['legend'] for _, s in entries[3:]]
ln('    ' + r' & '.join(r2) + r' \\')
ln(
    r'  \end{tabular}%',
    r'};',
    r'',
)

# ── Significance note ─────────────────────────────────────────────────────────
ln(
    r'\node[anchor=north, font=\scriptsize, text=black!60,',
    r'      text width=14cm, align=center] at',
    r'    ($(current bounding box.south)+(0,-1.45cm)$)',
    r'    {All pairwise comparisons vs.\ \textbf{Ours}:',
    r'     $p<0.001$ ($^{***}$, Wilcoxon signed-rank test).};',
    r'',
    r'\end{tikzpicture}',
    r'\end{document}',
)

# ── Write ─────────────────────────────────────────────────────────────────────
import subprocess, os
output = '\n'.join(L)
written_path = None
for path in OUT_PATHS:
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(output)
        print('Written: ' + path)
        written_path = path
    except OSError as e:
        print('Skipped: ' + path + '  (' + str(e) + ')')
print('Lines:   ' + str(len(L)))
if written_path:
    print("Compiling to PDF with XeLaTeX ...")
    subprocess.run(
        ["xelatex", "--enable-installer", "-interaction=nonstopmode",
         os.path.basename(written_path)],
        cwd=os.path.dirname(os.path.abspath(written_path)),
        check=True,
    )
    print("PDF ready: " + os.path.splitext(written_path)[0] + ".pdf")
