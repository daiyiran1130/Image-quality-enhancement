import pandas as pd
import re
import sys

# ── 配置：模型顺序、展示名、架构分组 ──────────────────────────────
MODEL_INFO = [
    # (parquet中的model_name, 展示名,                        架构组)
    ("LQ",         "LQ",                                    "Input"),
    ("Ours",       r"\textbf{Ours}",                        "Diffusion"),
    ("ResShift",   "ResShift (TPAMI, 2024)",                "Diffusion"),
    ("PowerPaint", "PowerPaint (ECCV, 2024)",               "Diffusion"),
    ("SD-v1.5",    "SD-v1.5 (CVPR, 2022)",                 "Diffusion"),
    ("DDPM",       "DDPM (NIPS, 2020)",                     "Diffusion"),
    ("RSCP2GAN",   "RSCP2GAN (TPAMI, 2025)",               "GAN"),
    ("MDA-Net",    r"MDA-Net (Med.\ Image Anal., 2024)",    "GAN"),
    ("OTE-GAN",    "OTE-GAN (ISBI, 2023)",                 "GAN"),
    ("Pix2PixGAN", "Pix2PixGAN (CVPR, 2017)",              "GAN"),
    ("PFT",        "PFT (CVPR, 2025)",                     "Trans."),
    ("RDSTN",      "RDSTN (ICASSP, 2024)",                 "Trans."),
    ("CMT",        "CMT (ICC, 2023)",                      "Trans."),
    ("GFE-Net",    "GFE-Net (MIA, 2023)",                  "Trans."),
]

METRICS = ["PSNR", "LPIPS", "VIF", "HARALICK"]
METRIC_HEADERS = [
    r"PSNR\,$\uparrow$",
    r"LPIPS\,$\downarrow$",
    r"VIF\,$\uparrow$",
    r"HARALICK\,$\downarrow$",
]

# ── 辅助函数 ──────────────────────────────────────────────────────
def parse_ci(ci_str):
    nums = re.findall(r"[\d.]+", str(ci_str))
    return float(nums[0]), float(nums[1])

def pstar_to_num(s):
    return {"***": 3, "**": 2, "*": 1}.get(str(s).strip(), 0)

def fmt(v):       return f"{float(v):.3f}"
def fmt_delta(v): v = float(v); return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

def statcell(val, ci_str, pstar_str, delta):
    lo, hi = parse_ci(ci_str)
    n = pstar_to_num(pstar_str)
    return (f"\\statcell{{{fmt(val)}}}"
            f"{{{fmt(lo)}\\text{{--}}{fmt(hi)}}}"
            f"{{{n}}}{{{fmt_delta(delta)}}}")

# ── LaTeX 前言 ────────────────────────────────────────────────────
PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{geometry}
\geometry{margin=1in, landscape}

% \pstarmark{n}: 0=空, 1=*, 2=†, 3=‡
\newcommand{\pstarmark}[1]{%
  \ifnum#1=1 {*}\fi%
  \ifnum#1=2 {\dag}\fi%
  \ifnum#1=3 {\ddag}\fi%
}

% \statcell{value}{CI}{stars}{delta}
\newcommand{\statcell}[4]{%
  \begin{tabular}[t]{@{}c@{}}
    $#1^{\,\pstarmark{#3}}_{\scriptstyle(#4)}$ \\[-3pt]
    {\scriptsize $#2$}
  \end{tabular}%
}

\begin{document}
"""

POSTAMBLE = r"\end{document}"

# ── 主函数 ────────────────────────────────────────────────────────
def generate(parquet_path: str, output_path: str = "table.tex"):
    df = pd.read_parquet(parquet_path).set_index("model_name")

    lines = [PREAMBLE]
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison of image enhancement methods. "
                 r"$\ddag$: $p<0.001$ vs.\ Ours; "
                 r"$\uparrow$higher is better; $\downarrow$lower is better.}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.6}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"  \toprule")
    lines.append("  " + " & ".join(["Arch.", "Model"] + METRIC_HEADERS) + " \\\\")
    lines.append(r"  \midrule")

    from itertools import groupby
    groups = [(arch, list(ms)) for arch, ms in groupby(MODEL_INFO, key=lambda x: x[2])]

    for g_idx, (arch, members) in enumerate(groups):
        n = len(members)
        for r_idx, (model_key, display_name, _) in enumerate(members):
            row = df.loc[model_key]
            cells = [statcell(row[m], row[f"{m}_95CI"], row[f"{m}_p_star"], row[f"{m}_delta"])
                     for m in METRICS]
            if r_idx == 0:
                arch_col = f"  \\multirow{{{n}}}{{*}}{{{arch}}}" if n > 1 else f"  {arch}"
                line = f"{arch_col}\n    & {display_name} &\n      "
            else:
                line = f"    & {display_name} &\n      "
            lines.append(line + " &\n      ".join(cells) + " \\\\")

        if g_idx < len(groups) - 1:
            lines.append(r"  \midrule")

    lines += [r"  \bottomrule", r"\end{tabular}", r"\end{table}", "", POSTAMBLE]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成：{output_path}")


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else "TASK_figure1.parquet"
    output  = sys.argv[2] if len(sys.argv) > 2 else "table.tex"
    generate(parquet, output)
