#!/usr/bin/env python3
"""
draw_figure2_simple.py — Simplified PSNR-classified bar chart.

Selects 5 representative models:
  Ours  |  SD-v1.5 (baseline)  |  best-other-Diffusion  |  best-GAN  |  best-Trans.

Selection is done automatically (image-count-weighted mean PSNR across PSNR scopes).

Layout : 4 rows (metrics) × 4 columns (PSNR scopes), 16 subplots.
Each subplot shows 5 bars with 95 % CI error bars and significance markers.
Optional connection lines link the same model across PSNR-scope columns.

Usage
-----
  python draw_figure2_simple.py [input.parquet] [output_stem]

Generates  <stem>_with_conn.tex / .pdf  and  <stem>_no_conn.tex / .pdf.
"""

import re
import sys
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Switches
# ══════════════════════════════════════════════════════════════════════════════

SHOW_CONNECTIONS: bool = True   # default; both versions always written in main()

LOCAL_PARQUET = r"D:\work\figure_table\figure2\classify_with_PSNR\classify_with_PSNR.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# Data configuration
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

ALWAYS_INCLUDE = {"Ours", "SD-v1.5"}   # always in the 5-model set
FAMILIES_TO_REPRESENT = ["Diffusion", "GAN", "Trans."]

PSNR_SCOPES = ["PSNR<20", "(20, 30)", "(30, 40)", "(40, 40+)"]
SCOPE_LABELS = [
    r"PSNR\,$<$\,20\ \ ($n=1547$)",
    r"PSNR\,$\in$\,(20,\,30)\ \ ($n=5894$)",
    r"PSNR\,$\in$\,(30,\,40)\ \ ($n=2108$)",
    r"PSNR\,$\geq$\,40\ \ ($n=1647$)",
]
SCOPE_COUNTS = {"PSNR<20": 1547, "(20, 30)": 5894, "(30, 40)": 2108, "(40, 40+)": 1647}

METRICS = ["PSNR", "LPIPS", "VIF", "HARALICK"]
METRIC_LABELS = [
    r"PSNR ($\uparrow$)",
    r"LPIPS ($\downarrow$)",
    r"VIF ($\uparrow$)",
    r"HARALICK ($\downarrow$, log)",
]
METRIC_LOGSCALE = {"PSNR": False, "LPIPS": False, "VIF": False, "HARALICK": True}

# Per-model visual style (colour + display name)
# Filled in dynamically after model selection
MODEL_DISPLAY = {
    "Ours":      r"\textbf{Ours}",
    "SD-v1.5":   r"SD-v1.5",
    "ResShift":  r"ResShift",
    "MDA-Net":   r"MDA-Net",
    "GFE-Net":   r"GFE-Net",
    # fallback for other possible selections
    "PowerPaint": r"PowerPaint",
    "DDPM":       r"DDPM",
    "RSCP2GAN":   r"RSCP2GAN",
    "OTE-GAN":    r"OTE-GAN",
    "Pix2PixGAN": r"Pix2PixGAN",
    "PFT":        r"PFT",
    "RDSTN":      r"RDSTN",
    "CMT":        r"CMT",
}

# ── subplot geometry (cm) ──────────────────────────────────────────────────
SUBPLOT_W = 10.0
SUBPLOT_H = 7.0
GAP_X     = 0.8
GAP_Y     = 1.8   # smaller gap: only 5 x-tick labels needed

# ══════════════════════════════════════════════════════════════════════════════
# Model selection
# ══════════════════════════════════════════════════════════════════════════════

def select_models(df_raw: pd.DataFrame) -> list[str]:
    """Return ordered list of 5 representative models."""
    total = sum(SCOPE_COUNTS.values())
    all_models = df_raw["model_name"].unique().tolist()

    # Image-count-weighted mean PSNR per model
    wavg: dict[str, float] = {}
    df_idx = df_raw.set_index(["PSNR_scope", "model_name"])
    for m in all_models:
        wavg[m] = sum(
            df_idx.loc[(s, m), "PSNR"] * SCOPE_COUNTS[s]
            for s in PSNR_SCOPES
            if (s, m) in df_idx.index
        ) / total

    selected: dict[str, str] = {}   # model_name → role label

    # Always-include models
    for m in ALWAYS_INCLUDE:
        selected[m] = ARCH_GROUP.get(m, "?")

    # Best from each family (excluding already-selected)
    for family in FAMILIES_TO_REPRESENT:
        candidates = {
            m: wavg[m]
            for m in all_models
            if ARCH_GROUP.get(m) == family and m not in selected
        }
        if candidates:
            best = max(candidates, key=candidates.get)
            selected[best] = family
            print(f"  Best {family} (excl. {ALWAYS_INCLUDE & set(m for m,g in ARCH_GROUP.items() if g==family)}): "
                  f"{best}  (PSNR={candidates[best]:.3f})")

    # Fixed display order: Ours first, then SD-v1.5, then others by PSNR desc
    others = sorted(
        [m for m in selected if m not in ALWAYS_INCLUDE],
        key=lambda m: wavg.get(m, 0), reverse=True
    )
    order = ["Ours", "SD-v1.5"] + others
    return [m for m in order if m in selected]


# ══════════════════════════════════════════════════════════════════════════════
# Per-model colour palette (pgfplots colour names)
# ══════════════════════════════════════════════════════════════════════════════

# Roles → colour name defined in preamble
# Same architecture→colour mapping as draw_figure2.py
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
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_ci(ci_str: str) -> tuple[float, float]:
    nums = re.findall(r"[\d.]+", str(ci_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return 0.0, 0.0


def load_data(parquet_path: str, models: list[str]) -> dict:
    df = pd.read_parquet(parquet_path).set_index(["PSNR_scope", "model_name"])
    out = {}
    for scope in PSNR_SCOPES:
        for model in models:
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


def yrange(data: dict, scope: str, metric: str, models: list[str],
           use_log: bool, n_brackets: int = 0) -> tuple[float, float]:
    vals_hi = [data[(scope, m, metric)]["hi"] for m in models if (scope, m, metric) in data]
    vals_lo = [data[(scope, m, metric)]["lo"] for m in models if (scope, m, metric) in data]
    if not vals_hi:
        return 0.0, 1.0
    if use_log:
        return 0.3, max(vals_hi) * (3.5 * (1.5 ** n_brackets) if n_brackets else 3.5)
    span = max(vals_hi) - min(vals_lo)
    ymin = max(0.0, min(vals_lo) - span * 0.04)
    ymax = max(vals_hi) + span * (0.22 + 0.14 * n_brackets)
    return ymin, ymax


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX generation
# ══════════════════════════════════════════════════════════════════════════════

def make_preamble() -> str:
    return r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,calc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\definecolor{myblue}  {RGB}{55, 119, 189}
\definecolor{myred}   {RGB}{210,  70,  70}
\definecolor{mygreen} {RGB}{ 50, 155,  70}
\definecolor{myorange}{RGB}{230, 130,  40}
\definecolor{mygray}  {RGB}{130, 130, 130}

\begin{document}"""


def generate_tex(data: dict, models: list[str], show_connections: bool) -> str:
    N = len(models)
    colours = [model_colour(m) for m in models]
    display = [MODEL_DISPLAY.get(m, m) for m in models]

    # Pre-scan: any *** significance across all subplots?
    has_triple_star_any = any(
        data.get((sc, m, me), {}).get("sig", "") == "***"
        for sc in PSNR_SCOPES for me in METRICS
        for m in models if m != "Ours"
    )

    # Bounding box
    total_w = len(PSNR_SCOPES) * SUBPLOT_W + (len(PSNR_SCOPES) - 1) * GAP_X
    total_h = (len(METRICS) - 1) * (SUBPLOT_H + GAP_Y) + SUBPLOT_H
    bb_left, bb_top    = -3.0,  2.0
    bb_right, bb_bottom = total_w + 0.5, -(total_h + 2.5 + (1.0 if has_triple_star_any else 0.0))

    lines = [make_preamble()]
    lines.append(r"\begin{tikzpicture}[font=\small]")
    lines.append(
        f"\\useasboundingbox ({bb_left:.1f}cm,{bb_top:.1f}cm)"
        f" rectangle ({bb_right:.1f}cm,{bb_bottom:.1f}cm);"
    )
    lines.append("")

    for row, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        use_log = METRIC_LOGSCALE[metric]

        for col, (scope, slabel) in enumerate(zip(PSNR_SCOPES, SCOPE_LABELS)):
            ax_x = col * (SUBPLOT_W + GAP_X)
            ax_y = -row * (SUBPLOT_H + GAP_Y)
            axname = f"ax{row}{col}"

            # Pre-compute significance brackets (*** omitted; shown in figure note)
            sig_brackets = []
            for _mi, _m in enumerate(models):
                if _m == "Ours":
                    continue
                _d = data.get((scope, _m, metric))
                if _d and _d["sig"] and _d["sig"] not in ("nan", "", "***"):
                    sig_brackets.append((_mi + 1, _d["sig"]))
            sig_brackets.sort(key=lambda t: t[0])

            ym, yM = yrange(data, scope, metric, models, use_log, len(sig_brackets))

            # ── begin axis ────────────────────────────────────────────────────
            lines += [
                f"% ── {metric} | {scope} ──────────────────────────────────",
                r"\begin{axis}[",
                f"  name={axname},",
                f"  at={{({ax_x:.2f}cm,{ax_y:.2f}cm)}},",
                f"  anchor=north west,",
                f"  width={SUBPLOT_W:.1f}cm,",
                f"  height={SUBPLOT_H:.1f}cm,",
                f"  ybar,",
                f"  bar width=14pt,",
                f"  xtick={{1,...,{N}}},",
            ]
            xlabels = ",".join("{" + display[i] + "}" for i in range(N))
            lines += [
                f"  xticklabels={{{xlabels}}},",
                f"  xticklabel style={{rotate=30, anchor=east, font=\\small}},",
                f"  ymin={ym:.4f},",
                f"  ymax={yM:.4f},",
                f"  xmin=0.2,",
                f"  xmax={N + 0.8},",
                f"  ylabel={{{mlabel}}},",
                f"  ylabel style={{font=\\footnotesize}},",
                f"  ymajorgrids=true,",
                f"  grid style={{dashed, gray!30}},",
                f"  tick style={{thin}},",
                f"  every tick label/.style={{font=\\footnotesize}},",
                f"  clip=false,",
                f"  scaled y ticks=false,",
                f"  y tick label style={{font=\\tiny,"
                f" /pgf/number format/fixed,"
                f" /pgf/number format/precision=3}},",
            ]
            if use_log:
                lines += ["  ymode=log,", "  log basis y={10},"]
            if row == 0:
                lines += [
                    f"  title={{{slabel}}},",
                    f"  title style={{font=\\small\\bfseries, align=center}},",
                ]
            lines.append(r"]")
            lines.append("")

            # ── one \addplot per model (individual colours) ────────────────
            for mi, (model, color) in enumerate(zip(models, colours)):
                d = data.get((scope, model, metric))
                if not d:
                    continue
                x = mi + 1
                lines += [
                    f"\\addplot[",
                    f"  ybar, fill={color}!70, draw={color}!90!black,"
                    f" line width=0.5pt,",
                    f"  error bars/.cd, y dir=both, y explicit,",
                    f"  error bar style={{line width=0.8pt, black}},",
                    f"] coordinates {{({x},{d['val']:.4f})"
                    f" +- ({d['err_lo']:.4f},{d['err_hi']:.4f})}};",
                ]

            lines.append("")

            # ── value labels (on top of bars), significance, coordinates ──
            for mi, (model, color) in enumerate(zip(models, colours)):
                d = data.get((scope, model, metric))
                if not d:
                    continue
                x = mi + 1

                # Numeric value label above the bar (3 decimal places)
                if use_log:
                    label_y = d["hi"] * 1.5
                else:
                    span = yM - ym
                    label_y = d["val"] + d["err_hi"] + span * 0.03

                lines.append(
                    f"\\node[above, font=\\tiny, inner sep=0pt,"
                    f" text={color}!70!black] "
                    f"at (axis cs:{x},{label_y:.4f}) {{{d['val']:.3f}}};"
                )

                # Named coordinate at bar top for connection lines
                if show_connections:
                    lines.append(
                        f"\\coordinate (C{row}S{col}M{mi})"
                        f" at (axis cs:{x},{d['val']:.4f});"
                    )

            # ── bracket-style significance markers (only * and **) ───────────
            if sig_brackets:
                all_hi_raw = [data[(scope, m, metric)]["hi"]
                              for m in models if (scope, m, metric) in data]
                all_lo_raw = [data[(scope, m, metric)]["lo"]
                              for m in models if (scope, m, metric) in data]
                if use_log:
                    max_hi = max(all_hi_raw)
                    base_y = max_hi * 1.5
                    step_f = 1.5
                    tick_r = 0.87
                    for level, (x2, sig_text) in enumerate(sig_brackets):
                        y_brk = base_y * (step_f ** level)
                        mid_x = (1.0 + x2) / 2.0
                        lines += [
                            f"\\draw[thin] (axis cs:1,{y_brk:.4f})"
                            f" -- (axis cs:{x2},{y_brk:.4f});",
                            f"\\draw[thin] (axis cs:1,{y_brk:.4f})"
                            f" -- (axis cs:1,{y_brk * tick_r:.4f});",
                            f"\\draw[thin] (axis cs:{x2},{y_brk:.4f})"
                            f" -- (axis cs:{x2},{y_brk * tick_r:.4f});",
                            f"\\node[above, font=\\tiny, inner sep=0pt]"
                            f" at (axis cs:{mid_x:.2f},{y_brk:.4f}) {{${sig_text}$}};",
                        ]
                else:
                    raw_span = max(all_hi_raw) - min(all_lo_raw)
                    base_y = max(all_hi_raw) + raw_span * 0.08
                    step   = raw_span * 0.14
                    tick_h = raw_span * 0.03
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
                            f"\\node[above, font=\\tiny, inner sep=0pt]"
                            f" at (axis cs:{mid_x:.2f},{y_brk:.4f}) {{${sig_text}$}};",
                        ]

            lines += [r"\end{axis}", ""]

    # ── Connection lines ──────────────────────────────────────────────────────
    if show_connections:
        lines.append("% ── Connection lines ─────────────────────────────────────────────")
        for row in range(len(METRICS)):
            for mi, model in enumerate(models):
                color = model_colour(model)
                lw    = "0.6pt"
                op    = "0.60"
                for col in range(len(PSNR_SCOPES) - 1):
                    c0 = f"C{row}S{col}M{mi}"
                    c1 = f"C{row}S{col+1}M{mi}"
                    lines.append(
                        f"\\draw[{color}!85!black, dashed, line width={lw},"
                        f" opacity={op}] ({c0}) -- ({c1});"
                    )
        lines.append("")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = -(total_h + 0.8)
    legend_parts = []
    for model, color in zip(models, colours):
        disp = MODEL_DISPLAY.get(model, model)
        legend_parts.append(
            rf"\tikz{{\fill[{color}!70] (0,0) rectangle (0.4cm,0.25cm);}}"
            rf"~{disp}"
        )
    legend_content = r"\quad ".join(legend_parts)
    lines += [
        f"% ── Legend ─────────────────────────────────────────────────────────",
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
    parquet = sys.argv[1] if len(sys.argv) > 1 else LOCAL_PARQUET
    stem    = sys.argv[2] if len(sys.argv) > 2 else "figure2_simple"

    print(f"Reading: {parquet}")
    df_raw = pd.read_parquet(parquet)

    print("Selecting representative models …")
    models = select_models(df_raw)
    print(f"Selected ({len(models)}): {models}")

    data = load_data(parquet, models)
    print(f"Loaded {len(data)} data cells.")

    for show_conn in [True, False]:
        suffix   = "with_conn" if show_conn else "no_conn"
        out_path = f"{stem}_{suffix}.tex"
        tex = generate_tex(data, models, show_conn)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(tex)
        print(f"Written: {out_path}")

    print()
    print("Compile with:")
    print(f"  pdflatex {stem}_with_conn.tex")
    print(f"  pdflatex {stem}_no_conn.tex")


if __name__ == "__main__":
    main()
