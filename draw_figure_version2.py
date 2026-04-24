#!/usr/bin/env python3
"""
draw_figure_version2.py — Box-plot comparison of 5 representative models.

Each box shows the distribution of metric values across the 4 PSNR quality
scopes (PSNR<20, 20-30, 30-40, ≥40), with whiskers at min/max.
A diamond marker (◆) indicates the image-count-weighted mean.
Significance annotations compare each model to Ours using overall
statistics from TASK_figure1.parquet.

Layout  : 2 columns × 2 rows  (PSNR | LPIPS / VIF | HARALICK)
Models  : Ours · SD-v1.5 · ResShift (best-Diff.) · MDA-Net (best-GAN) · GFE-Net (best-Trans.)
Output  : fig1.1.2.tex

Usage
-----
  python draw_figure_version2.py [classify.parquet] [overall.parquet]
"""

import os
import re
import sys
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Paths (Windows local paths; relative fallbacks used in CI)
# ══════════════════════════════════════════════════════════════════════════════

LOCAL_CLASSIFY = r"D:\work\figure_table\figure2\classify_with_PSNR\classify_with_PSNR.parquet"
LOCAL_OVERALL  = r"D:\work\figure_table\figure2\TASK_figure1.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# Shared colour + architecture config  (identical to draw_figure2*.py)
# ══════════════════════════════════════════════════════════════════════════════

ARCH_GROUP = {
    "LQ":          "Input",
    "Ours":        "Diffusion", "ResShift":    "Diffusion",
    "PowerPaint":  "Diffusion", "SD-v1.5":     "Diffusion", "DDPM": "Diffusion",
    "RSCP2GAN":    "GAN",       "MDA-Net":     "GAN",
    "OTE-GAN":     "GAN",       "Pix2PixGAN":  "GAN",
    "PFT":         "Trans.",    "RDSTN":       "Trans.",
    "CMT":         "Trans.",    "GFE-Net":     "Trans.",
    "Pre+SK":      "Pre+",      "Pre+CFG":     "Pre+",
    "Pre+Seg":     "Pre+",      "Pre+CFG+Seg": "Pre+",
    "Pre+SK+CFG":  "Pre+",      "Pre+SK+Seg":  "Pre+",
}

ARCH_COLORS = {
    "Input":     "mygray",
    "Diffusion": "myblue",
    "GAN":       "myred",
    "Trans.":    "mygreen",
    "Pre+":      "myorange",
}

def model_colour(model: str) -> str:
    return ARCH_COLORS.get(ARCH_GROUP.get(model, "?"), "mygray")

# ══════════════════════════════════════════════════════════════════════════════
# Data configuration
# ══════════════════════════════════════════════════════════════════════════════

PSNR_SCOPES  = ["PSNR<20", "(20, 30)", "(30, 40)", "(40, 40+)"]
SCOPE_COUNTS = {"PSNR<20": 1547, "(20, 30)": 5894, "(30, 40)": 2108, "(40, 40+)": 1647}

METRICS = ["PSNR", "LPIPS", "VIF", "HARALICK"]
METRIC_LABELS = [
    r"PSNR ($\uparrow$)",
    r"LPIPS ($\downarrow$)",
    r"VIF ($\uparrow$)",
    r"HARALICK ($\downarrow$)",
]

MODEL_DISPLAY = {
    "Ours":     r"\textbf{Ours}",
    "SD-v1.5":  r"SD-v1.5",
    "ResShift": r"ResShift",
    "MDA-Net":  r"MDA-Net",
    "GFE-Net":  r"GFE-Net",
}

ALWAYS_INCLUDE       = {"Ours", "SD-v1.5"}
FAMILIES_TO_REPRESENT = ["Diffusion", "GAN", "Trans."]

# subplot geometry (cm)
SUBPLOT_W = 8.0
SUBPLOT_H = 6.5
GAP_X     = 1.8   # horizontal gap between subplots
GAP_Y     = 2.2   # vertical gap (space for rotated x-tick labels)
GRID_COLS = 2     # 2×2 layout

# ══════════════════════════════════════════════════════════════════════════════
# Model selection  (same logic as draw_figure2_simple.py)
# ══════════════════════════════════════════════════════════════════════════════

def select_models(df_classify: pd.DataFrame) -> list[str]:
    total      = sum(SCOPE_COUNTS.values())
    all_models = df_classify["model_name"].unique().tolist()
    df_idx     = df_classify.set_index(["PSNR_scope", "model_name"])
    wavg = {
        m: sum(
            df_idx.loc[(s, m), "PSNR"] * SCOPE_COUNTS[s]
            for s in PSNR_SCOPES if (s, m) in df_idx.index
        ) / total
        for m in all_models
    }
    selected: set[str] = set(ALWAYS_INCLUDE)
    for family in FAMILIES_TO_REPRESENT:
        candidates = {
            m: wavg[m] for m in all_models
            if ARCH_GROUP.get(m) == family and m not in selected
        }
        if candidates:
            best = max(candidates, key=candidates.get)
            selected.add(best)
            print(f"  Best {family}: {best}  (wPSNR={candidates[best]:.3f})")
    others = sorted(selected - ALWAYS_INCLUDE, key=lambda m: wavg.get(m, 0), reverse=True)
    return ["Ours", "SD-v1.5"] + others

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_scope_values(classify_path: str, models: list[str]) -> dict:
    """For each (model, metric): list of mean values across PSNR scopes."""
    df    = pd.read_parquet(classify_path).set_index(["PSNR_scope", "model_name"])
    total = sum(SCOPE_COUNTS.values())
    out   = {}
    for model in models:
        for metric in METRICS:
            vals, wvals = [], []
            for scope in PSNR_SCOPES:
                if (scope, model) in df.index:
                    v = float(df.loc[(scope, model), metric])
                    vals.append(v)
                    wvals.append(v * SCOPE_COUNTS[scope])
            if vals:
                out[(model, metric)] = {
                    "scope_vals": vals,
                    "wmean": sum(wvals) / total,
                }
    return out


def load_significance(overall_path: str, classify_path: str,
                      models: list[str]) -> dict[tuple, str]:
    """Load overall p_star per (model, metric) vs Ours.
    Tries overall_path first; falls back to (20,30)-scope from classify."""
    # Try TASK_figure1.parquet (several candidate locations)
    candidates = [
        overall_path,
        os.path.join(os.path.dirname(os.path.abspath(classify_path)), "TASK_figure1.parquet"),
        "TASK_figure1.parquet",
    ]
    for path in candidates:
        try:
            df  = pd.read_parquet(path).set_index("model_name")
            out = {}
            for model in models:
                if model == "Ours":
                    continue
                for metric in METRICS:
                    col = f"{metric}_p_star"
                    if col in df.columns and model in df.index:
                        out[(model, metric)] = str(df.loc[model, col]).strip()
            if out:
                print(f"  Significance loaded from: {path}")
                return out
        except Exception:
            pass

    # Fallback: use the scope with most images
    print("  Significance: falling back to (20,30) scope from classify parquet")
    df  = pd.read_parquet(classify_path).set_index(["PSNR_scope", "model_name"])
    out = {}
    for model in models:
        if model == "Ours":
            continue
        for metric in METRICS:
            try:
                out[(model, metric)] = str(
                    df.loc[("(20, 30)", model), f"{metric}_p_star"]
                ).strip()
            except KeyError:
                pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Box statistics
# ══════════════════════════════════════════════════════════════════════════════

def box_stats(vals: list[float]) -> dict:
    """5-number summary + mean from a list of observations (one per PSNR scope)."""
    arr = np.array(vals, dtype=float)
    q1  = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lw  = float(max(arr.min(), q1 - 1.5 * iqr))
    uw  = float(min(arr.max(), q3 + 1.5 * iqr))
    return {"lw": lw, "q1": q1, "med": med, "q3": q3, "uw": uw}

# ══════════════════════════════════════════════════════════════════════════════
# LaTeX / pgfplots generation
# ══════════════════════════════════════════════════════════════════════════════

PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,calc}
\usepackage{pgfplots}
\usepgfplotslibrary{statistics}
\pgfplotsset{compat=1.18}

%% ── Colours (identical to draw_figure2*.py) ─────────────────────────────────
\definecolor{myblue}  {RGB}{55, 119, 189}
\definecolor{myred}   {RGB}{210,  70,  70}
\definecolor{mygreen} {RGB}{ 50, 155,  70}
\definecolor{myorange}{RGB}{230, 130,  40}
\definecolor{mygray}  {RGB}{130, 130, 130}

\begin{document}"""


def generate_tex(scope_data: dict, sig: dict, models: list[str]) -> str:
    N       = len(models)
    n_rows  = (len(METRICS) + GRID_COLS - 1) // GRID_COLS   # = 2
    colours = [model_colour(m) for m in models]
    display = [MODEL_DISPLAY.get(m, m) for m in models]

    # Compute all box stats upfront
    stats: dict[tuple, dict] = {}
    for model in models:
        for metric in METRICS:
            k = (model, metric)
            if k in scope_data:
                stats[k] = box_stats(scope_data[k]["scope_vals"])

    # Pre-scan: any *** significance?
    has_triple_star_any = any(
        sig.get((m, me), "") == "***"
        for me in METRICS for m in models if m != "Ours"
    )

    # Bounding box for standalone
    total_w = GRID_COLS * SUBPLOT_W + (GRID_COLS - 1) * GAP_X
    total_h = (n_rows - 1) * (SUBPLOT_H + GAP_Y) + SUBPLOT_H
    extra_bottom = 1.0 if has_triple_star_any else 0.0
    bb = (f"\\useasboundingbox (-3.0cm,1.8cm)"
          f" rectangle ({total_w + 0.5:.1f}cm,{-(total_h + 2.5 + extra_bottom):.1f}cm);")

    lines = [PREAMBLE, r"\begin{tikzpicture}[font=\small]", bb, ""]

    for idx, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        row = idx // GRID_COLS
        col = idx  % GRID_COLS
        ax_x   = col * (SUBPLOT_W + GAP_X)
        ax_y   = -row * (SUBPLOT_H + GAP_Y)
        axname = f"ax{row}{col}"

        # y-axis range from all box stats for this metric
        all_lw = [stats[(m, metric)]["lw"] for m in models if (m, metric) in stats]
        all_uw = [stats[(m, metric)]["uw"] for m in models if (m, metric) in stats]
        if not all_lw:
            continue
        span = max(all_uw) - min(all_lw)
        ymin = max(0.0, min(all_lw) - span * 0.06)

        # Pre-compute significance brackets (*** omitted; shown in figure note)
        sig_brackets = []
        for _mi, _m in enumerate(models):
            if _m == "Ours":
                continue
            _p = sig.get((_m, metric), "")
            if _p and _p not in ("nan", "", "***"):
                sig_brackets.append((_mi + 1, _p))
        sig_brackets.sort(key=lambda t: t[0])

        ymax = max(all_uw) + span * (0.10 + 0.18 * len(sig_brackets))

        xlabels_str = ",".join("{" + display[i] + "}" for i in range(N))

        # ── begin axis ────────────────────────────────────────────────────────
        lines += [
            f"% ── {metric} (row={row}, col={col}) ─────────────────────────────",
            r"\begin{axis}[",
            f"  name={axname},",
            f"  at={{({ax_x:.2f}cm,{ax_y:.2f}cm)}},",
            f"  anchor=north west,",
            f"  width={SUBPLOT_W:.1f}cm,",
            f"  height={SUBPLOT_H:.1f}cm,",
            f"  boxplot/draw direction=y,",
            f"  boxplot/box extend=0.32,",
            f"  xtick={{1,...,{N}}},",
            f"  xticklabels={{{xlabels_str}}},",
            f"  xticklabel style={{rotate=25, anchor=east, font=\\small}},",
            f"  ymin={ymin:.4f}, ymax={ymax:.4f},",
            f"  ylabel={{{mlabel}}},",
            f"  ylabel style={{font=\\footnotesize}},",
            f"  ymajorgrids=true,",
            f"  grid style={{dashed, gray!30}},",
            f"  tick style={{thin}},",
            f"  every tick label/.style={{font=\\footnotesize}},",
            f"  clip=false,",
            f"  scaled y ticks=false,",
            f"  y tick label style={{font=\\tiny,"
            f" /pgf/number format/fixed, /pgf/number format/precision=3}},",
            f"  title={{{mlabel}}},",
            f"  title style={{font=\\small\\bfseries}},",
            r"]",
            "",
        ]

        # ── box plots ─────────────────────────────────────────────────────────
        for mi, (model, color) in enumerate(zip(models, colours)):
            s = stats.get((model, metric))
            if not s:
                continue
            wmean = scope_data[(model, metric)]["wmean"]
            lines += [
                f"% {model}",
                r"\addplot[",
                f"  boxplot prepared={{",
                f"    lower whisker={s['lw']:.4f},",
                f"    lower quartile={s['q1']:.4f},",
                f"    median={s['med']:.4f},",
                f"    upper quartile={s['q3']:.4f},",
                f"    upper whisker={s['uw']:.4f},",
                f"    average={wmean:.4f},",
                f"  }},",
                f"  fill={color}!50, draw={color}!85!black, line width=0.7pt,",
                f"  boxplot/every average/.style={{mark=diamond*,"
                f" mark size=2.5pt, fill={color}, draw={color}!70!black}},",
                r"] coordinates {};",
            ]

        lines.append("")

        # ── bracket-style significance markers ────────────────────────────────
        if sig_brackets:
            all_uw_vals = [stats[(m, metric)]["uw"]
                           for m in models if (m, metric) in stats]
            base_y = max(all_uw_vals) + span * 0.06
            step   = span * 0.17
            tick_h = span * 0.04
            for level, (x2, sig_text) in enumerate(sig_brackets):
                y_brk = base_y + level * step
                mid_x = (1.0 + x2) / 2.0
                lines += [
                    f"\\draw[thin] (axis cs:1,{y_brk:.4f})"
                    f" -- (axis cs:{x2},{y_brk:.4f});",
                    f"\\draw[thin] (axis cs:1,{y_brk:.4f})"
                    f" -- (axis cs:1,{y_brk - tick_h:.4f});",
                    f"\\draw[thin] (axis cs:{x2},{y_brk:.4f})"
                    f" -- (axis cs:{x2},{y_brk - tick_h:.4f});",
                    f"\\node[above, font=\\small, inner sep=0pt]"
                    f" at (axis cs:{mid_x:.2f},{y_brk:.4f}) {{${sig_text}$}};",
                ]

        lines += [r"\end{axis}", ""]

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = -(total_h + 0.9)
    parts = [
        rf"\tikz{{\fill[{c}!50] (0,0) rectangle (0.35cm,0.22cm);}}"
        rf"~{MODEL_DISPLAY.get(m, m)}"
        for m, c in zip(models, colours)
    ]
    lines += [
        f"\\node[anchor=north west, font=\\small]"
        f" at (0cm,{legend_y:.2f}cm) {{",
        "  " + r"\quad ".join(parts),
        "};",
        "",
    ]

    if has_triple_star_any:
        note_y = legend_y - 0.75
        lines += [
            f"\\node[anchor=north west, font=\\footnotesize, text=gray!80!black]"
            f" at (0cm,{note_y:.2f}cm) {{",
            r"  $^{***}$\,$p<0.001$ vs.\ Ours: all annotated comparisons reach"
            r" this significance level and are not individually marked.",
            "};",
            "",
        ]

    lines += [r"\end{tikzpicture}", "", r"\end{document}"]
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    classify_path = sys.argv[1] if len(sys.argv) > 1 else LOCAL_CLASSIFY
    overall_path  = sys.argv[2] if len(sys.argv) > 2 else LOCAL_OVERALL
    out_path      = "fig1.1.2.tex"

    print(f"Reading: {classify_path}")
    df_classify = pd.read_parquet(classify_path)

    print("Selecting models …")
    models = select_models(df_classify)
    print(f"Selected: {models}")

    scope_data = load_scope_values(classify_path, models)
    sig        = load_significance(overall_path, classify_path, models)
    print(f"Loaded {len(scope_data)} (model, metric) entries, "
          f"{len(sig)} significance entries.")

    tex = generate_tex(scope_data, sig, models)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"Written: {out_path}")
    print(f'\nCompile with:  pdflatex "{out_path}"')


if __name__ == "__main__":
    main()
