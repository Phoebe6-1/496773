import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill

# ====== 读入系数对比表 ======
df = pd.read_csv("output_beta_compare/coef_compare.csv")   # <- 改成你的路径

# 你脚本里常见的列名（若不同，下面两行改一下即可）
TERM_COL = "term"
DIFF_COL = "coef_diff_fans_minus_judges"   # Fans - Judges
ABS_COL  = "abs_diff"

# ====== 1) 去掉 season_ 相关项（以及截距） ======
df[TERM_COL] = df[TERM_COL].astype(str)
mask = (~df[TERM_COL].str.startswith("season_")) & (df[TERM_COL] != "Intercept") & (df[TERM_COL] != "Precision")
df2 = df.loc[mask].copy()

# ====== 2) 取 Top-K（在“非season”里重新排序） ======
TOPK = 12
df2 = df2.sort_values(ABS_COL, ascending=False).head(TOPK)

# 为了画图更顺眼：按差值从小到大排序（负的在下、正的在上）
df2 = df2.sort_values(DIFF_COL, ascending=True)

# ====== 3) 美化标签（可按需修改映射） ======
def pretty_name(s: str) -> str:
    s = s.replace("industry_grp_", "Industry: ")
    s = s.replace("ProStrength", "Pro Strength")
    s = s.replace("ProCount", "Pro Count")
    s = s.replace("age_z2", "Age (z^2)")
    s = s.replace("age_z", "Age (z)")
    s = s.replace("US", "Home: US")
    # 标签过长就自动换行
    return fill(s, width=18)

df2["label"] = df2[TERM_COL].apply(pretty_name)

# ====== 4) 画“更美观”的横向柱状图 ======
vals = df2[DIFF_COL].values

# 正负两色（更易读，也更“论文图”）
colors = np.where(vals >= 0, "#4C78A8", "#F58518")   # 蓝/橙

fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=220)

ax.barh(df2["label"], vals, color=colors, alpha=0.92)
ax.axvline(0, color="black", linewidth=1.0)

# 让左右留白对称（更好看）
m = np.max(np.abs(vals)) if len(vals) else 1.0
ax.set_xlim(-1.15*m, 1.15*m)

ax.set_title("Top Features with Largest Effect Differences (excluding Season)", fontsize=14, pad=10)
ax.set_xlabel("Coefficient Difference (Fans - Judges)", fontsize=12)

# 网格 + 去掉上/右边框
ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 在柱子末端标注数值（可选，但通常更清晰）
for y, v in enumerate(vals):
    ax.text(v + (0.01*m if v >= 0 else -0.01*m),
            y,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10)

plt.tight_layout()
plt.savefig("top_diff_bar_no_season.png", bbox_inches="tight")
plt.show()
