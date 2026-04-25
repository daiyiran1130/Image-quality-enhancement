#!/usr/bin/env python3
"""
draw_figure2.py — Generate LaTeX/pgfplots bar-chart figure for
PSNR-classified image quality metrics.

Layout : 4 rows (metrics) × 4 columns (PSNR scopes), 16 subplots.
Each subplot shows all 20 models as bars (coloured by architecture),
with 95 % CI error bars and significance markers.
Optional connection lines link the same model across PSNR-scope columns.

Usage
-----
  python draw_figure2.py [input.parquet] [output_stem]

If called with no arguments:
  - reads  LOCAL_PARQUET  (Windows local path)
  - writes  <stem>.tex  (连线由 SHOW_CONNECTIONS 开关控制)
"""

import re
import sys
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# User-facing switches
# ══════════════════════════════════════════════════════════════════════════════

SHOW_CONNECTIONS: bool = True   # ← 在这里改：True = 画连线，False = 不画

LOCAL_PARQUET = r"D:\work\figure_table\figure2\classify_with_PSNR\classify_with_PSNR.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# Data configuration
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ORDER = [
    "LQ",
    "Ours", "ResShift", "PowerPaint", "SD-v1.5", "DDPM",
    "RSCP2GAN", "MDA-Net", "OTE-GAN", "Pix2PixGAN",
    "PFT", "RDSTN", "CMT", "GFE-Net",
    "Pre+SK", "Pre+CFG", "Pre+Seg", "Pre+CFG+Seg", "Pre+SK+CFG", "Pre+SK+Seg",
]

MODEL_DISPLAY = {
    "LQ":          r"LQ",
    "Ours":        r"\textbf{Ours}",
    "ResShift":    r"ResShift",
    "PowerPaint":  r"PowerPaint",
    "SD-v1.5":     r"SD-v1.5",
    "DDPM":        r"DDPM",
    "RSCP2GAN":    r"RSCP2GAN",
    "MDA-Net":     r"MDA-Net",
    "OTE-GAN":     r"OTE-GAN",
    "Pix2PixGAN":  r"Pix2PixGAN",
    "PFT":         r"PFT",
    "RDSTN":       r"RDSTN",
    "CMT":         r"CMT",
    "GFE-Net":     r"GFE-Net",
    "Pre+SK":      r"Pre$+$SK",
    "Pre+CFG":     r"Pre$+$CFG",
    "Pre+Seg":     r"Pre$+$Seg",
    "Pre+CFG+Seg": r"Pre$+$CFG$+$Seg",
    "Pre+SK+CFG":  r"Pre$+$SK$+$CFG",
    "Pre+SK+Seg":  r"Pre$+$SK$+$Seg",
}

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

ARCH_LIST   = ["Input", "Diffusion", "GAN", "Trans.", "Pre+"]
ARCH_COLORS = {
    "Input":     "mygray",
    "Diffusion": "myblue",
    "GAN":       "myred",
    "Trans.":    "mygreen",
    "Pre+":      "myorange",
}

PSNR_SCOPES = ["PSNR<20", "(20, 30)", "(30, 40)", "(40, 40+)"]
SCOPE_LABELS = [
    r"PSNR\,$<$\,20\ \ ($n=1547$)",
    r"PSNR\,$\in$\,(20,\,30)\ \ ($n=5894$)",
    r"PSNR\,$\in$\,(30,\,40)\ \ ($n=2108$)",
    r"PSNR\,$\geq$\,40\ \ ($n=1647$)",
]

METRICS = ["PSNR", "LPIPS", "VIF", "HARALICK"]
METRIC_LABELS = [
    r"PSNR ($\uparrow$)",
    r"LPIPS ($\downarrow$)",
    r"VIF ($\uparrow$)",
    r"HARALICK ($\downarrow$, log)",
]
METRIC_LOGSCALE = {"PSNR": False, "LPIPS": False, "VIF": False, "HARALICK": True}

SUBPLOT_W  = 13.5
SUBPLOT_H  = 7.0
GAP_X      = 0.6
GAP_Y      = 2.8

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_ci(ci_str):
    nums = re.findall(r"[\d.]+", str(ci_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return 0.0, 0.0


def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    df = df.set_index(["PSNR_scope", "model_name"])
    out = {}
    for scope in PSNR_SCOPES:
        for model in MODEL_ORDER:
            for metric in METRICS:
                try:
                    row = df.loc[(scope, model)]
                    val = float(row[metric])
                    lo, hi = parse_ci(row[f"{metric}_95CI"])
                    sig = str(row[f"{metric}_p_star"]).strip()
                    out[(scope, model, metric)] = {
                        "val": val, "lo": lo, "hi": hi, "sig": sig,
                        "err_lo": val - lo, "err_hi": hi - val,
                    }
                except KeyError:
                    pass
    return out


def yrange(data, scope, metric, use_log):
    vals_hi = [data[(scope, m, metric)]["hi"] for m in MODEL_ORDER if (scope, m, metric) in data]
    vals_lo = [data[(scope, m, metric)]["lo"] for m in MODEL_ORDER if (scope, m, metric) in data]
    if not vals_hi:
        return 0.0, 1.0
    if use_log:
        return 0.3, max(vals_hi) * 3.5
    span = max(vals_hi) - min(vals_lo)
    ymin = max(0.0, min(vals_lo) - span * 0.04)
    ymax = max(vals_hi) + span * 0.20
    return ymin, ymax


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX generation
# ══════════════════════════════════════════════════════════════════════════════

PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage{fontspec}
\setmainfont{Arial}
\setsansfont{Arial}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,calc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

%% ── Architecture group colours ─────────────────────────────────────────────
\definecolor{myblue}  {RGB}{55, 119, 189}
\definecolor{myred}   {RGB}{210,  70,  70}
\definecolor{mygreen} {RGB}{ 50, 155,  70}
\definecolor{myorange}{RGB}{230, 130,  40}
\definecolor{mygray}  {RGB}{130, 130, 130}

\begin{document}"""

POSTAMBLE = r"""
\end{document}"""


def generate_tex(data, show_connections: bool) -> str:
    has_triple_star_any = any(
        data.get((sc, m, me), {}).get("sig", "") == "***"
        for sc in PSNR_SCOPES for me in METRICS
        for m in MODEL_ORDER
    )

    total_w = len(PSNR_SCOPES) * SUBPLOT_W + (len(PSNR_SCOPES) - 1) * GAP_X
    total_h = len(METRICS) * SUBPLOT_H + (len(METRICS) - 1) * GAP_Y
    bb_left   = -3.0
    bb_top    = 2.0
    bb_right  = total_w + 0.5
    bb_bottom = -(total_h + 2.5 + (1.0 if has_triple_star_any else 0.0))

    lines = [PREAMBLE]
    lines.append(r"\begin{tikzpicture}[font=\small]")
    lines.append(
        f"\\useasboundingbox ({bb_left:.1f}cm,{bb_top:.1f}cm)"
        f" rectangle ({bb_right:.1f}cm,{bb_bottom:.1f}cm);"
    )
    lines.append("")

    N = len(MODEL_ORDER)

    arch_members: dict[str, list[tuple[int, str]]] = {a: [] for a in ARCH_LIST}
    for mi, model in enumerate(MODEL_ORDER):
        arch_members[ARCH_GROUP[model]].append((mi, model))

    for row, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        use_log = METRIC_LOGSCALE[metric]

        for col, (scope, slabel) in enumerate(zip(PSNR_SCOPES, SCOPE_LABELS)):
            ax_x = col * (SUBPLOT_W + GAP_X)
            ax_y = -row * (SUBPLOT_H + GAP_Y)
            axname = f"ax{row}{col}"

            ym, yM = yrange(data, scope, metric, use_log)

            lines += [
                f"% ── {metric} | {scope} ────────────────────────────",
                r"\begin{axis}[",
                f"  name={axname},",
                f"  at={{({ax_x:.2f}cm,{ax_y:.2f}cm)}},",
                f"  anchor=north west,",
                f"  width={SUBPLOT_W:.1f}cm,",
                f"  height={SUBPLOT_H:.1f}cm,",
                f"  ybar,",
                f"  bar width=4pt,",
                f"  xtick={{1,...,{N}}},",
            ]

            xlabels = ",".join("{" + MODEL_DISPLAY[m] + "}" for m in MODEL_ORDER)
            lines += [
                f"  xticklabels={{{xlabels}}},",
                f"  xticklabel style={{rotate=50, anchor=east, font=\\tiny}},",
                f"  ymin={ym:.4f},",
                f"  ymax={yM:.4f},",
                f"  xmin=0.2,",
                f"  xmax={N + 0.8},",
                f"  ylabel={{{mlabel}}},",
                f"  ylabel style={{font=\\footnotesize}},",
                f"  ymajorgrids=true,",
                f"  grid style={{dashed, gray!30}},",
                f"  tick style={{thin}},",
                f"  every tick label/.style={{font=\\tiny}},",
                f"  clip=false,",
                f"  scaled y ticks=false,",
                f"  y tick label style={{font=\\tiny, /pgf/number format/fixed,"
                f"  /pgf/number format/precision=3}},",
            ]
            if use_log:
                lines += [f"  ymode=log,", f"  log basis y={{10}},"]
            if row == 0:
                lines += [
                    f"  title={{{slabel}}},",
                    f"  title style={{font=\\small\\bfseries, align=center}},",
                ]
            lines.append(r"]")
            lines.append("")

            for arch in ARCH_LIST:
                members = arch_members[arch]
                color   = ARCH_COLORS[arch]
                coords  = []
                for (mi, model) in members:
                    d = data.get((scope, model, metric))
                    if d:
                        x = mi + 1
                        coords.append(
                            f"({x},{d['val']:.4f}) +- ({d['err_lo']:.4f},{d['err_hi']:.4f})"
                        )
                if not coords:
                    continue
                lines += [
                    f"\\addplot[",
                    f"  ybar,",
                    f"  fill={color}!65, draw={color}!85!black, line width=0.3pt,",
                    f"  error bars/.cd, y dir=both, y explicit,",
                    f"  error bar style={{line width=0.6pt, black}},",
                    f"] coordinates {{",
                    "  " + " ".join(coords),
                    "};",
                ]

            lines.append("")

            for mi, model in enumerate(MODEL_ORDER):
                d = data.get((scope, model, metric))
                if not d:
                    continue
                x = mi + 1

                if d["sig"] and d["sig"] not in ("nan", "", "***"):
                    if use_log:
                        sig_y = d["hi"] * 1.6
                    else:
                        span = yM - ym
                        sig_y = d["hi"] + span * 0.02
                    lines.append(
                        f"\\node[above, font=\\tiny, inner sep=0pt] "
                        f"at (axis cs:{x},{sig_y:.4f}) {{${d['sig']}$}};"
                    )

                if show_connections:
                    lines.append(
                        f"\\coordinate (C{row}S{col}M{mi}) "
                        f"at (axis cs:{x},{d['val']:.4f});"
                    )

            lines += [r"\end{axis}", ""]

    if show_connections:
        lines.append("% ── Connection lines ──────────────────────────────────────────────")
        for row in range(len(METRICS)):
            for mi, model in enumerate(MODEL_ORDER):
                color = ARCH_COLORS[ARCH_GROUP[model]]
                for col in range(len(PSNR_SCOPES) - 1):
                    c0 = f"C{row}S{col}M{mi}"
                    c1 = f"C{row}S{col+1}M{mi}"
                    lines.append(
                        f"\\draw[{color}!80!black, dashed, line width=0.5pt,"
                        f" opacity=0.60] ({c0}) -- ({c1});"
                    )
        lines.append("")

    bottom_of_last_row = (len(METRICS) - 1) * (SUBPLOT_H + GAP_Y) + SUBPLOT_H
    legend_y = -(bottom_of_last_row + 0.8)
    arch_legend_parts = []
    for arch in ARCH_LIST:
        color = ARCH_COLORS[arch]
        arch_legend_parts.append(
            rf"\tikz{{\fill[{color}!65] (0,0) rectangle (0.35cm,0.22cm);}}"
            rf"~\textbf{{{arch}}}"
        )
    legend_content = r"\quad ".join(arch_legend_parts)
    lines += [
        f"\\node[anchor=north west, font=\\small] at (0cm,{legend_y:.2f}cm) {{",
        f"  {legend_content}",
        "};",
        "",
    ]

    if has_triple_star_any:
        note_y = legend_y - 0.75
        lines += [
            f"\\node[anchor=north west, font=\\footnotesize, text=gray!80!black]"
            f" at (0cm,{note_y:.2f}cm) {{",
            r"  $^{***}$\,$p<0.001$ vs.\ Ours: all comparisons reach this significance level and are not individually annotated.",
            "};",
            "",
        ]

    lines += [r"\end{tikzpicture}", "", POSTAMBLE]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parquet = sys.argv[1] if len(sys.argv) > 1 else LOCAL_PARQUET
    stem    = sys.argv[2] if len(sys.argv) > 2 else "Draw fig1.1.1complex"

    print(f"Reading: {parquet}")
    data = load_data(parquet)
    print(f"Loaded {len(data)} data cells.")

    out_path = f"{stem}.tex"
    tex = generate_tex(data, SHOW_CONNECTIONS)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"Written: {out_path}  (连线={'开' if SHOW_CONNECTIONS else '关'})")
    import subprocess, os
    print("Compiling to PDF with XeLaTeX ...")
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", os.path.basename(out_path)],
        cwd=os.path.dirname(os.path.abspath(out_path)),
        check=True,
    )
    print(f"PDF ready: {os.path.splitext(out_path)[0]}.pdf")


if __name__ == "__main__":
    main()
