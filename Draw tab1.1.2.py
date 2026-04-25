import pandas as pd
import re
import sys
from itertools import groupby

MODEL_INFO = [
    ("LQ",         "LQ",                                     "Input"),
    ("Ours",       r"\textbf{Ours}",                         "Diffusion"),
    ("ResShift",   "ResShift (TPAMI, 2024)",                 "Diffusion"),
    ("PowerPaint", "PowerPaint (ECCV, 2024)",                "Diffusion"),
    ("SD-v1.5",    "SD-v1.5 (CVPR, 2022)",                  "Diffusion"),
    ("DDPM",       "DDPM (NIPS, 2020)",                      "Diffusion"),
    ("RSCP2GAN",   "RSCP2GAN (TPAMI, 2025)",                "GAN"),
    ("MDA-Net",    r"MDA-Net (Med.\ Image Anal., 2024)",     "GAN"),
    ("OTE-GAN",    "OTE-GAN (ISBI, 2023)",                  "GAN"),
    ("Pix2PixGAN", "Pix2PixGAN (CVPR, 2017)",               "GAN"),
    ("PFT",        "PFT (CVPR, 2025)",                      "Trans."),
    ("RDSTN",      "RDSTN (ICASSP, 2024)",                  "Trans."),
    ("CMT",        "CMT (ICC, 2023)",                       "Trans."),
    ("GFE-Net",    "GFE-Net (MIA, 2023)",                   "Trans."),
]

METRICS = ["PSNR", "LPIPS", "VIF", "HARALICK"]
METRIC_HEADERS = [
    r"PSNR\,$\uparrow$",
    r"LPIPS\,$\downarrow$",
    r"VIF\,$\uparrow$",
    r"HARALICK\,$\downarrow$",
]

FAMILIES = {
    "Diffusion": ["ResShift", "PowerPaint", "SD-v1.5", "DDPM"],
    "GAN":       ["RSCP2GAN", "MDA-Net", "OTE-GAN", "Pix2PixGAN"],
    "Trans.":    ["PFT", "RDSTN", "CMT", "GFE-Net"],
}

def parse_ci(ci_str):
    nums = re.findall(r"[\d.]+", str(ci_str))
    return float(nums[0]), float(nums[1])

def pstar_to_num(s):
    return {"***": 3, "**": 2, "*": 1}.get(str(s).strip(), 0)

def fmt(v):       return f"{float(v):.3f}"
def fmt_delta(v): v = float(v); return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{geometry}
\usepackage[table]{xcolor}
\geometry{margin=1in}

\definecolor{bestrow}{rgb}{0.80, 0.91, 0.97}

\newcommand{\pstarmark}[1]{%
  \ifnum#1=1 {*}\fi%
  \ifnum#1=2 {\dag}\fi%
  \ifnum#1=3 {\ddag}\fi%
}

% \bestcell{value}{stars}{delta} — 最强对比模型专用
\newcommand{\bestcell}[3]{%
  $#1^{\,\pstarmark{#2}}_{\scriptstyle(#3)}$%
}

\begin{document}
"""

POSTAMBLE = r"\end{document}"

def generate_simple(parquet_path: str, output_path: str = "table_simple.tex"):
    df = pd.read_parquet(parquet_path).set_index("model_name")

    best_models = {
        fam: max(models, key=lambda m: df.loc[m, "PSNR"])
        for fam, models in FAMILIES.items()
    }

    ours = df.loc["Ours"]
    ci_parts = []
    for m in METRICS:
        lo, hi = parse_ci(ours[f"{m}_95CI"])
        ci_parts.append(
            f"{m}\\,=\\,{fmt(ours[m])} (95\\%\\,CI: {fmt(lo)}--{fmt(hi)})"
        )
    caption = (
        "Simplified quantitative comparison. "
        "The strongest comparison model per architecture family (by PSNR) is \\colorbox{bestrow}{highlighted}; "
        "subscript values show absolute difference relative to Ours "
        "($\\ddag$: $p<0.001$ vs.\\ Ours). "
        "Ours achieves " + ", ".join(ci_parts) + ". "
        "Full 95\\% confidence intervals and significance analyses for all other models "
        "are provided in the Appendix. "
        "All comparison models showed statistically significant differences "
        "from Ours ($p < 0.001$)."
    )

    lines = [PREAMBLE]
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.4}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"  \toprule")
    lines.append("  " + " & ".join(["Arch.", "Model"] + METRIC_HEADERS) + " \\\\")
    lines.append(r"  \midrule")

    groups = [(arch, list(ms)) for arch, ms in groupby(MODEL_INFO, key=lambda x: x[2])]

    for g_idx, (arch, members) in enumerate(groups):
        best_key = best_models.get(arch)
        nrows    = len(members)

        for r_idx, (model_key, display_name, _) in enumerate(members):
            row     = df.loc[model_key]
            is_best = (model_key == best_key)

            cells = []
            for m in METRICS:
                val = fmt(row[m])
                if is_best:
                    n_star = pstar_to_num(row[f"{m}_p_star"])
                    delta  = fmt_delta(row[f"{m}_delta"])
                    cells.append(f"\\bestcell{{{val}}}{{{n_star}}}{{{delta}}}")
                else:
                    cells.append(f"${val}$")

            # 高亮行从第二列开始加颜色，第一列（multirow）保持不变
            if is_best:
                colored_name = r"\cellcolor{bestrow} " + display_name
                cell_str = " & ".join(r"\cellcolor{bestrow} " + c for c in cells)
            else:
                colored_name = display_name
                cell_str = " & ".join(cells)

            if r_idx == 0:
                arch_col = f"\\multirow{{{nrows}}}{{*}}{{{arch}}}"
            else:
                arch_col = ""

            lines.append(f"  {arch_col} & {colored_name} & {cell_str} \\\\")

        if g_idx < len(groups) - 1:
            lines.append(r"  \midrule")

    lines += [r"  \bottomrule", r"\end{tabular}", r"\end{table}", "", POSTAMBLE]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成：{output_path}")


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\work\figure_table\figure1\TASK_figure1.parquet"
    output  = sys.argv[2] if len(sys.argv) > 2 else "Draw tab1.1.2.tex"
    generate_simple(parquet, output)