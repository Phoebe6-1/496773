import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("output_sim/controversy_summary.csv")
rank = df[["celebrity_name", "tag", "placement_sim"]].copy()

# --------- 顺序（可选，但建议：让 x 轴更“有逻辑”）---------
# 按最好的名次（越小越靠前）排序 celebrity
cele_order = (
    rank.groupby("celebrity_name")["placement_sim"]
        .min()
        .sort_values()
        .index
        .tolist()
)

# 固定 tag 顺序（保证颜色/图例稳定）
tag_order = ["percent", "percent_bottom2", "rank", "rank_bottom2"]
rank["celebrity_name"] = pd.Categorical(rank["celebrity_name"], categories=cele_order, ordered=True)
rank["tag"] = pd.Categorical(rank["tag"], categories=tag_order, ordered=True)

# --------- 画图风格：更论文、更干净 ---------
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 马卡龙配色 + marker（你也可以按喜好改）
palette = {
    "percent": "#8FBBD9",          # 柔和蓝
    "percent_bottom2": "#5F9EA0",  # 柔和蓝绿
    "rank": "#F4A7B9",             # 柔和粉
    "rank_bottom2": "#B7E4C7",     # 柔和薄荷绿
}
markers = {
    "percent": "o",
    "percent_bottom2": "X",
    "rank": "s",
    "rank_bottom2": "P",
}

# --------- 核心：给不同 method 做 x 偏移（避免重叠）---------
celebs = rank["celebrity_name"].cat.categories.tolist()
x_base = np.arange(len(celebs))

offsets = {
    "percent": -0.24,
    "percent_bottom2": -0.08,
    "rank": +0.08,
    "rank_bottom2": +0.24,
}

for tag in tag_order:
    sub = rank[rank["tag"] == tag]
    # 把类别 celebrity 映射到 x 坐标
    x = sub["celebrity_name"].cat.codes.values + offsets[tag]

    ax.scatter(
        x, sub["placement_sim"].values,
        s=150,
        marker=markers[tag],
        color=palette[tag],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.95,
        label=tag.replace("_", "\n")
    )

# --------- 轴与网格 ---------
ax.invert_yaxis()
ymax = int(rank["placement_sim"].max())
ax.set_yticks(range(1, ymax + 1))
ax.set_ylim(ymax + 0.6, 0.4)

ax.set_xticks(x_base)
ax.set_xticklabels(celebs, rotation=20, ha="right")

ax.grid(axis="y", alpha=0.25)
ax.grid(axis="x", visible=False)
sns.despine(ax=ax, left=False, bottom=False)

ax.set_title("Comparison of Contestant Placements by Simulation Method", pad=14)
ax.set_xlabel("")  # x 轴已经是名字，留空更干净
ax.set_ylabel("Simulated Placement (1st = Top)")

# 图例放外面，不遮挡数据
ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

plt.tight_layout()
plt.show()
